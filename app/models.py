import os
from functools import lru_cache
from ultralytics import YOLO
from .config import settings


def _model_file(kind: str, variant: str) -> str:
    base = settings.BASE_MODELS_DIR

    if kind == "lane":
        # LANE: ƯU TIÊN .pt (segment chuẩn như Colab), rồi mới onnx / torchscript
        candidates = [
            (os.path.join(base, kind, variant, f"{variant}.pt"),         ".pt"),
            (os.path.join(base, kind, variant, variant),                 "no-ext (.pt)"),
            (os.path.join(base, kind, variant, f"{variant}.onnx"),       ".onnx"),
            (os.path.join(base, kind, variant, f"{variant}.torchscript"),".torchscript"),
        ]
        for p, label in candidates:
            if os.path.exists(p):
                print(f"[models] Using {label} model for {kind}/{variant}: {p}")
                return p

    else:
        # SIGN: GIỮ Y NGUYÊN – ưu tiên ONNX, rồi TorchScript
        p_onnx = os.path.join(base, kind, variant, f"{variant}.onnx")
        if os.path.exists(p_onnx):
            print(f"[models] Using ONNX model for {kind}/{variant}: {p_onnx}")
            return p_onnx

        p_ts = os.path.join(base, kind, variant, f"{variant}.torchscript")
        if os.path.exists(p_ts):
            print(f"[models] Using TorchScript model for {kind}/{variant}: {p_ts}")
            return p_ts

    raise FileNotFoundError(f"Không tìm thấy model cho {kind}/{variant} trong {base}")


@lru_cache(maxsize=8)
def get_model(kind: str, variant: str) -> YOLO:
    mp = _model_file(kind, variant)
    print(f"[models] Loading model from: {mp}")
    m = YOLO(mp)
    print(f"[models] Model loaded. Model type: {type(m)}")
    return m


def warmup_models():
    try:
        get_model("sign", "best")
    except Exception as e:
        print("Skip preload sign:", e)
    try:
        get_model("lane", "best")
    except Exception as e:
        print("Skip preload lane:", e)
