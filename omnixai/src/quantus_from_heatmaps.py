# mit python run_all_quantus.py ausführen (um alle Modelle und Methoden zu testen)
# Dieses Skript lädt Heatmaps aus Ordnern, nur von Resnet-Modellen werden MPRT-Metriken berechnet.
import argparse
from pathlib import Path
import gc
import random
import numpy as np
import torch
from PIL import Image
import pandas as pd
import quantus
import torch.nn as nn
import torchvision.transforms as T
from omnixai.data.image import Image as OmniImage
from omnixai.explainers.vision.specific.gradcam.gradcam import GradCAM, GradCAMPlus
from omnixai.explainers.vision.specific.ig import IntegratedGradientImage
from omnixai.explainers.vision.agnostic.lime import LimeImage
from src import configs
from src.model_def import build_model


def load_images_with_labels(data_dir, img_size, limit=None, paths_override=None):
    data_dir = Path(data_dir)

    if paths_override is None:
        paths = sorted(
            list(data_dir.glob("*.png"))
            + list(data_dir.glob("*.jpg"))
            + list(data_dir.glob("*.jpeg"))
        )
    else:
        paths = list(paths_override)

    if limit is not None:
        paths = paths[:limit]

    if not paths:
        raise RuntimeError(f"Keine Bilder in {data_dir} gefunden.")

    class_to_idx = {name: idx for idx, name in enumerate(configs.CLASSES)}

    labels_csv = data_dir / "labels.csv"
    if labels_csv.exists():
        df = pd.read_csv(labels_csv)
        file_to_class = dict(zip(df["file"], df["class"]))

        labels = []
        for p in paths:
            cls_name = file_to_class.get(p.name)
            if cls_name is None:
                raise RuntimeError(f"Keine Klasse für {p.name} in {labels_csv}")
            if cls_name not in class_to_idx:
                raise RuntimeError(
                    f"Klasse '{cls_name}' aus labels.csv nicht in configs.CLASSES: {list(class_to_idx.keys())}"
                )
            labels.append(class_to_idx[cls_name])
    else:
        labels = []
        for p in paths:
            cls_name = p.parent.name
            if cls_name not in class_to_idx:
                raise RuntimeError(
                    f"Ordnername '{cls_name}' nicht in configs.CLASSES: {list(class_to_idx.keys())}"
                )
            labels.append(class_to_idx[cls_name])

    imgs = []
    for p in paths:
        img = Image.open(p).convert("L")
        img = img.resize((img_size, img_size))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.repeat(arr[..., None], 3, axis=2)

        mean = np.array(configs.IMAGENET_MEAN, dtype=np.float32)
        std = np.array(configs.IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std

        imgs.append(arr.transpose(2, 0, 1))

    X = np.stack(imgs)
    y = np.array(labels, dtype=np.int64)
    return X, y, paths


def to_2d_heatmap_from_omnixai(explanation_data) -> np.ndarray:
    if isinstance(explanation_data, np.ndarray):
        heat = explanation_data
    elif isinstance(explanation_data, dict):
        if "scores" in explanation_data:
            heat = explanation_data["scores"]
        elif "data" in explanation_data:
            heat = explanation_data["data"]
        elif "importances" in explanation_data:
            heat = explanation_data["importances"]
        elif "masks" in explanation_data:
            m = explanation_data["masks"]
            heat = m[0] if isinstance(m, (list, tuple)) else m
        else:
            raise ValueError(
                f"Unbekanntes OmniXAI-Dict-Format: keys={list(explanation_data.keys())}"
            )
    else:
        raise ValueError(f"Unbekanntes OmniXAI-Format: type={type(explanation_data)}")

    heat = np.asarray(heat)

    if heat.ndim == 3:
        if heat.shape[-1] == 1:
            heat = heat[..., 0]
        elif heat.shape[0] == 1:
            heat = heat[0]
        else:
            heat = heat.mean(axis=-1) if heat.shape[-1] in (3, 4) else heat.mean(axis=0)

    if heat.ndim != 2:
        raise ValueError(f"Heatmap nicht 2D, shape={heat.shape}")

    heat = heat.astype(np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat = heat - heat.min()
    mx = heat.max()
    if mx > 1e-8:
        heat = heat / mx
    else:
        h, w = heat.shape
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        heat = (yy + xx) * 1e-8
    if np.allclose(heat.std(), 0.0):
        h, w = heat.shape
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        heat = heat + (yy + xx) * 1e-8
    return heat

def make_omnixai_explain_func(method: str, preprocess_fn, last_conv_getter):
    """
    Speedups:
      (1) Cache OmniXAI explainer objects per (model_id, method) so Quantus metrics
          calling explain_func repeatedly don't rebuild explainers every time.
      (2) Convert the whole batch from normalized CHW float -> uint8 HWC in one go,
          then only do lightweight PIL construction per sample.
    """
    explainer_cache = {}  # key: (id(model), method) -> explainer
    call_counter = {"n": 0}

    # Precompute mean/std once (broadcast-ready)
    mean = np.array(configs.IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(configs.IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    def _get_or_create_explainer(model):
        key = (id(model), method)
        if key in explainer_cache:
            return explainer_cache[key]

        model.eval()

        if method == "gradcam":
            explainer = GradCAM(
                model=model,
                target_layer=last_conv_getter(model),
                preprocess_function=preprocess_fn,
                mode="classification",
            )
        elif method == "gradcam++":
            explainer = GradCAMPlus(
                model=model,
                target_layer=last_conv_getter(model),
                preprocess_function=preprocess_fn,
                mode="classification",
            )
        elif method == "ig":
            explainer = IntegratedGradientImage(
                model=model,
                preprocess_function=preprocess_fn,
                mode="classification",
            )
        elif method == "lime":
            # predict_fn must close over THIS model instance
            def predict_fn(batch: OmniImage):
                with torch.no_grad():
                    t = preprocess_fn(batch)
                    logits = model(t)
                    probs = torch.softmax(logits, dim=1)
                return probs.detach().cpu().numpy()

            explainer = LimeImage(predict_function=predict_fn, mode="classification")
        else:
            raise ValueError(f"Unknown method: {method}")

        explainer_cache[key] = explainer
        return explainer

    def explain_func(*args, **kwargs):
        call_counter["n"] += 1

        # Optional logging controls (avoid slowing down metrics)
        verbose = bool(kwargs.get("verbose", False))
        prefix = kwargs.get("progress_prefix", "EXPLAIN")
        log_every = int(kwargs.get("log_every", 5))

        model = kwargs.get("model", None)
        x = kwargs.get("x_batch", None)
        y = kwargs.get("y_batch", None)

        if x is None:
            x = kwargs.get("inputs", None)
        if x is None:
            x = kwargs.get("x", None)

        if y is None:
            y = kwargs.get("targets", None)
        if y is None:
            y = kwargs.get("y", None)

        if model is None and len(args) >= 1:
            model = args[0]
        if x is None and len(args) >= 2:
            x = args[1]
        if y is None and len(args) >= 3:
            y = args[2]

        if model is None or x is None:
            raise ValueError(
                f"explain_func: missing model/x. Got keys={list(kwargs.keys())}, len(args)={len(args)}"
            )

        if verbose:
            print(f"[XAI] explain_func call #{call_counter['n']} (method={method}, batch={len(x)})")

        explainer = _get_or_create_explainer(model)

        # ---- SPEEDUP (2): batch denorm + uint8 conversion once ----
        # Expect x in shape (N, 3, H, W) and float range roughly normalized
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected x as (N,3,H,W), got shape={x.shape}")

        x_dn = x * std + mean                         # (N,3,H,W)
        x_dn = np.clip(x_dn, 0.0, 1.0)
        x_u8 = (x_dn.transpose(0, 2, 3, 1) * 255.0)    # (N,H,W,3)
        x_u8 = x_u8.astype(np.uint8, copy=False)

        attrs = []

        # LIME reproducibility: set seeds once per call (not per sample)
        if method == "lime":
            random.seed(configs.SEED)
            np.random.seed(configs.SEED)
            torch.manual_seed(configs.SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(configs.SEED)

        for i in range(x_u8.shape[0]):
            if verbose and (i % log_every) == 0:
                print(f"[{prefix}] explaining sample {i+1}/{x_u8.shape[0]}")

            pil = Image.fromarray(x_u8[i])  # cheap now
            omni = OmniImage(pil, batched=False)

            if method == "lime":
                exp = explainer.explain(omni, hide_color=0, num_samples=800)
            else:
                exp = explainer.explain(omni)

            explanation_data = exp.get_explanations()[0]
            heat2d = to_2d_heatmap_from_omnixai(explanation_data)
            attrs.append(heat2d[None, ...])

        return np.stack(attrs)

    return explain_func

# def make_omnixai_explain_func(method: str, preprocess_fn, last_conv_getter):
#     call_counter = {"n": 0}
#     def explain_func(*args, **kwargs):
#         call_counter["n"] += 1
#         print(f"[XAI] explain_func call #{call_counter['n']}")
#         prefix = kwargs.get("progress_prefix", "EXPLAIN")
#         log_every = int(kwargs.get("log_every", 5)) 
#         model = kwargs.get("model", None)
#         x = kwargs.get("x_batch", None)
#         y = kwargs.get("y_batch", None)

#         if x is None:
#             x = kwargs.get("inputs", None)
#         if x is None:
#             x = kwargs.get("x", None)

#         if y is None:
#             y = kwargs.get("targets", None)
#         if y is None:
#             y = kwargs.get("y", None)
#         if model is None and len(args) >= 1:
#             model = args[0]
#         if x is None and len(args) >= 2:
#             x = args[1]
#         if y is None and len(args) >= 3:
#             y = args[2]

#         if model is None or x is None:
#             raise ValueError(
#                 f"explain_func: missing model/x. Got keys={list(kwargs.keys())}, "
#                 f"len(args)={len(args)}"
#             )

#         model.eval()
#         if method == "gradcam":
#             explainer = GradCAM(
#                 model=model,
#                 target_layer=last_conv_getter(model),
#                 preprocess_function=preprocess_fn,
#                 mode="classification",
#             )
#         elif method == "gradcam++":
#             explainer = GradCAMPlus(
#                 model=model,
#                 target_layer=last_conv_getter(model),
#                 preprocess_function=preprocess_fn,
#                 mode="classification",
#             )
#         elif method == "ig":
#             explainer = IntegratedGradientImage(
#                 model=model,
#                 preprocess_function=preprocess_fn,
#                 mode="classification",
#             )
#         elif method == "lime":

#             def predict_fn(batch: OmniImage):
#                 with torch.no_grad():
#                     t = preprocess_fn(batch)
#                     logits = model(t)
#                     probs = torch.softmax(logits, dim=1)
#                 return probs.detach().cpu().numpy()

#             explainer = LimeImage(predict_function=predict_fn, mode="classification")
#         else:
#             raise ValueError(f"Unknown method: {method}")

#         attrs = []
#         mean = np.array(configs.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
#         std = np.array(configs.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

#         for i in range(len(x)):
#             if (i % log_every) == 0:
#                 print(f"[{prefix}] explaining sample {i+1}/{len(x)}")
#             img = x[i]
#             img_dn = img * std + mean
#             img_dn = np.clip(img_dn, 0.0, 1.0)
#             pil = Image.fromarray((img_dn.transpose(1, 2, 0) * 255).astype(np.uint8))
#             omni = OmniImage(pil, batched=False)

#             if method == "lime":
#                 random.seed(configs.SEED)
#                 np.random.seed(configs.SEED)
#                 torch.manual_seed(configs.SEED)
#                 if torch.cuda.is_available():
#                     torch.cuda.manual_seed_all(configs.SEED)
#                 exp = explainer.explain(omni, hide_color=0, num_samples=800)
#             else:
#                 exp = explainer.explain(omni)

#             explanation_data = exp.get_explanations()[0]
#             heat2d = to_2d_heatmap_from_omnixai(explanation_data)
#             attrs.append(heat2d[None, ...])

#         return np.stack(attrs)

#     return explain_func


def stratified_indices(y, n_total=30, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    n_per = int(np.ceil(n_total / len(classes)))

    idxs = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        idxs.extend(c_idx[:n_per].tolist())

    rng.shuffle(idxs)
    return np.array(idxs[:n_total], dtype=int)


def pair_by_stem(img_paths, hm_paths):
    img_map = {p.stem: p for p in img_paths}
    hm_map = {p.stem: p for p in hm_paths}

    common = sorted(set(img_map.keys()) & set(hm_map.keys()))
    if not common:
        raise RuntimeError(
            "Keine gemeinsamen IDs zwischen Bildern und Heatmaps gefunden."
        )

    imgs = [img_map[k] for k in common]
    hms = [hm_map[k] for k in common]
    return imgs, hms, common


def load_heatmaps(hm_dir, img_size, limit=None, paths_override=None):
    hm_dir = Path(hm_dir)

    paths = (
        sorted(list(hm_dir.glob("*.npy")))
        if paths_override is None
        else list(paths_override)
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"Keine Heatmaps in {hm_dir} gefunden.")

    hmaps = []
    for p in paths:
        arr = np.load(p).astype(np.float32)

        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                arr = arr.mean(axis=-1)

        if arr.ndim != 2:
            raise ValueError(f"Heatmap {p.name} hat unerwartetes shape={arr.shape}")

        arr = arr - arr.min()
        mx = arr.max()
        if mx > 1e-8:
            arr = arr / mx
        if arr.shape != (img_size, img_size):
            img = Image.fromarray((arr * 255).astype(np.uint8))
            img = img.resize((img_size, img_size))
            arr = np.array(img).astype(np.float32) / 255.0

        arr = np.clip(arr, 0.0, 1.0)
        hmaps.append(arr[None, ...])

    return np.stack(hmaps), paths


def find_last_conv(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Keine Conv2d-Layer im Modell gefunden.")
    return last


# _TFM = T.Compose(
#     [
#         T.Resize((configs.IMG_SIZE, configs.IMG_SIZE)),
#         T.Grayscale(num_output_channels=3),
#         T.ToTensor(),
#         T.Normalize(mean=configs.IMAGENET_MEAN, std=configs.IMAGENET_STD),
#     ]
# )


def make_preprocess(device: str):
    def preprocess_fn(batch: OmniImage):
        tensors = []
        for img in batch:
            pil = img.to_pil().convert("RGB")
            pil = pil.resize((configs.IMG_SIZE, configs.IMG_SIZE))
            t = T.ToTensor()(pil)
            t = T.Normalize(mean=configs.IMAGENET_MEAN, std=configs.IMAGENET_STD)(t)
            tensors.append(t)
        return torch.stack(tensors).to(device)
    return preprocess_fn


def load_model(model_name, num_classes, ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict):
        state = blob.get("state_dict", blob)
    else:
        state = blob

    model = build_model(
        model_name, num_classes=num_classes, pretrained=configs.PRETRAINED
    )
    clean_state = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--method",
        required=True,
        choices=[
            "gradcam", 
            "gradcam++", 
            "ig", 
            "lime"
            ],
    )
    ap.add_argument("--data_dir", default="data/ecg_test_flat")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--mprt_samples", type=int, default=30)
    args = ap.parse_args()
    img_size = configs.IMG_SIZE
    num_classes = len(configs.CLASSES)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] IMG_SIZE: {img_size}")

    hm_dir = Path("outputs/explanations") / args.model / args.method / "heatmap"
    if not hm_dir.exists():
        print(f"[ERROR] Heatmap-Ordner fehlt: {hm_dir}")
        return

    print(f"[INFO] Lade Heatmap-Pfade aus {hm_dir} ...")
    _, hm_paths = load_heatmaps(hm_dir, img_size=img_size, limit=None)

    print(f"[INFO] Lade Bild-Pfade aus {args.data_dir} ...")
    _, _, img_paths_all = load_images_with_labels(
        args.data_dir, img_size=img_size, limit=None
    )

    img_paths_paired, hm_paths_paired, ids = pair_by_stem(img_paths_all, hm_paths)
    if args.limit is not None:
        img_paths_paired = img_paths_paired[: args.limit]
        hm_paths_paired = hm_paths_paired[: args.limit]
        ids = ids[: args.limit]

    print(f"[INFO] Gepaarte Samples: {len(ids)}")
    X, Y, _ = load_images_with_labels(
        args.data_dir, img_size=img_size, limit=None, paths_override=img_paths_paired
    )
    H, _ = load_heatmaps(
        hm_dir, img_size=img_size, limit=None, paths_override=hm_paths_paired
    )

    ckpt_path = f"outputs/checkpoints/{args.model}_best.pt"
    print(f"[INFO] Lade Modell: {ckpt_path}")
    model = load_model(
        args.model, num_classes=num_classes, ckpt_path=ckpt_path, device=device
    )
    print("[DEBUG] First paired IDs:", ids[:5])
    print(
        "[DEBUG] First img:",
        img_paths_paired[0].name,
        "First hm:",
        hm_paths_paired[0].name,
    )

    metrics = {
        "faithfulness_corr": quantus.FaithfulnessCorrelation(
            nr_runs=50,
            subset_size=50,
            disable_warnings=True,
        ),
        "max_sens": quantus.MaxSensitivity(
            nr_samples=7,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            similarity_func=quantus.similarity_func.difference,
            abs=True,
            normalise=True,
            disable_warnings=True,
        ),
    }

    ENABLE_MPRT_MODELS = {"resnet18", "resnet34", "resnet50", "resnet101", "densenet121", "densenet169", "vgg16_bn",  "efficientnet_b0", "efficientnet_b1", "mobilenet_v2"}
    explain_func = None

    preprocess_fn = make_preprocess(device=device)

    def last_conv_getter(m):
        if args.model == "densenet121":
            return m.features.denseblock4.denselayer16.conv2
        if args.model == "densenet169":
            return m.features.denseblock4.denselayer32.conv2
        return find_last_conv(m)

    explain_func = make_omnixai_explain_func(
        method=args.method,
        preprocess_fn=preprocess_fn,
        last_conv_getter=last_conv_getter,
    )
    if args.model in ENABLE_MPRT_MODELS:
    # if args.model in ENABLE_MPRT_MODELS and args.method != "lime":
        metrics["mprt"] = quantus.MPRT(
            return_average_correlation=True,
            disable_warnings=True,
        )

    scores = {}

    print("[INFO] Running metric: faithfulness_corr (GPU) ...")
    vals = metrics["faithfulness_corr"](
        model=model,
        x_batch=X,
        y_batch=Y,
        a_batch=H,
        device=device,
    )
    scores["faithfulness_corr"] = float(
        np.mean(list(vals.values()) if isinstance(vals, dict) else vals)
    )
    print(f"[INFO] faithfulness_corr: mean = {scores['faithfulness_corr']:.6f}")

    print("[INFO] Running metric: max_sens (GPU) ...")
    vals = metrics["max_sens"](
        model=model,
        x_batch=X,
        y_batch=Y,
        a_batch=None,
        device=device,
        explain_func=explain_func,
        explain_func_kwargs={"progress_prefix": "MaxSens", "log_every": 5},
    )

    scores["max_sens"] = float(
        np.mean(list(vals.values()) if isinstance(vals, dict) else vals)
    )
    print(f"[INFO] max_sens: mean = {scores['max_sens']:.6f}")

    if "mprt" in metrics:
        mprt_idx = stratified_indices(Y, n_total=args.mprt_samples, seed=0)

        X_m = X[mprt_idx]
        Y_m = Y[mprt_idx]

        print(
            f"[INFO] Running metric: mprt (GPU) on {len(mprt_idx)} stratified samples ..."
        )

        vals = metrics["mprt"](
            model=model,
            x_batch=X_m,
            y_batch=Y_m,
            a_batch=None,
            device=device,
            explain_func=explain_func,
            explain_func_kwargs={"progress_prefix": "MPRT", "log_every": 1},
        )
        mprt_vals = np.array(list(vals.values()) if isinstance(vals, dict) else vals, dtype=float)
        scores["mprt"] = float(np.nanmean(mprt_vals))
        print(f"[INFO] mprt: mean = {scores['mprt']:.6f}")
    else:
        scores["mprt"] = float("nan")

    out_dir = Path("outputs/metrics/quantus_raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"quantus_{args.model}_{args.method}.csv"

    df = pd.DataFrame(
        [
            {
                "model": args.model,
                "method": args.method,
                "faithfulness_corr": scores["faithfulness_corr"],
                "max_sens": scores["max_sens"],
                "mprt": scores["mprt"],
            }
        ]
    )
    df.to_csv(out_file, index=False)

    print("\n[RESULTS]")
    print(df)
    print(f"[OK] Quantus-Ergebnisse gespeichert in: {out_file}")


if __name__ == "__main__":
    main()
