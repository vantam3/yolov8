import os
import cv2
import asyncio
import numpy as np
from PIL import Image
from fastapi import UploadFile
from ultralytics import YOLO

from .models import get_model
from .config import settings
from .utils import (
    read_upload_image,
    pil_to_bytes,
    np_bgr_to_pil,
    save_upload_to_temp_video,
    ensure_fps,
    video_writer,
    as_mp4_bytes,
    draw_yolo_predictions,
)


def ensure_bgr_ndarray(x):
    """
    Đảm bảo input là numpy BGR (H, W, 3) để dùng chung cho np_bgr_to_pil.
    Hỗ trợ: np.ndarray (BGR/GRAY/BGRA) hoặc PIL.Image.
    """
    # Trường hợp đã là numpy
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            # GRAY -> BGR
            return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        if x.ndim == 3 and x.shape[2] == 4:
            # BGRA -> BGR
            return cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
        return x

    # Trường hợp là PIL.Image
    if isinstance(x, Image.Image):
        arr = np.array(x)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            # RGBA -> RGB
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        # RGB -> BGR
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr

    raise TypeError(f"Unsupported image type for ensure_bgr_ndarray: {type(x)}")


# ========== IMAGE ==========
async def infer_lane_image(image: UploadFile, variant: str, conf: float):
    model: YOLO = get_model("lane", variant)

    # Đọc ảnh & ép về numpy BGR
    raw_img = await read_upload_image(image)
    img_bgr = ensure_bgr_ndarray(raw_img)

    used_conf = conf if conf is not None else 0.35

    res = model.predict(
        source=img_bgr,
        imgsz=getattr(settings, "LANE_IMGSZ", 960),   
        conf=used_conf,
        iou=0.45,
        max_det=100,
        retina_masks=True,   # cần cho segment
        save=False,
        verbose=False,
    )[0]

    # --- debug ---
    try:
        n_boxes = int(res.boxes.shape[0]) if res.boxes is not None else 0
    except Exception:
        n_boxes = 0

    try:
        n_masks = 0 if (res.masks is None or res.masks.data is None) else int(res.masks.data.shape[0])
    except Exception:
        n_masks = 0

    print(
        f"[lane:image] boxes={n_boxes}, masks={n_masks}, "
        f"conf={used_conf}, provider={os.environ.get('ORT_PROVIDERS','')}"
    )

    if res.masks is not None and res.masks.data is not None and n_masks > 0:
        m0 = res.masks.data[0]
        print(
            f"[lane:image] mask[0] shape={m0.shape}, "
            f"min={float(m0.min())}, max={float(m0.max())}"
        )
        print("[lane:image] Drawing masks (res.plot)")
        out_bgr = res.plot()
        # Nếu chỉ muốn mask:
        # out_bgr = res.plot(labels=False, boxes=False, probs=False)
    else:
        print("[lane:image] No masks → trả về ảnh gốc")
        out_bgr = img_bgr

    out_pil = np_bgr_to_pil(out_bgr)
    return pil_to_bytes(out_pil, fmt="JPEG", quality=90)


# ========== VIDEO ==========
# ========== VIDEO ==========
async def infer_lane_video(video: UploadFile, variant: str, conf: float, stride: int):
    model: YOLO = get_model("lane", variant)
    src_path = await save_upload_to_temp_video(video)

    print(f"[lane:video] ===== START PROCESS VIDEO =====")
    print(f"[lane:video] Input path: {src_path}")

    cap = cv2.VideoCapture(src_path)
    fps = ensure_fps(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"[lane:video] size={w}x{h}, fps={fps}, total_frames={total_frames}, stride={stride}")

    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Không đọc được video input.")

    out_path = src_path + ".out.webm"
    writer = video_writer(out_path, w, h, fps)
    print(f"[lane:video] Output path: {out_path}")

    idx = 0
    used_conf = conf if conf is not None else 0.35
    last_result = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[lane:video] Hết frame, dừng đọc video.")
            break

        # log mỗi 30 frame (sau skip stride) cho đỡ spam
        if idx % 30 == 0:
            if total_frames > 0:
                pct = 100.0 * idx / total_frames
                print(f"[lane:video] ... processing frame {idx}/{total_frames} (~{pct:.1f}%)")
            else:
                print(f"[lane:video] ... processing frame {idx}")

        # skip frame theo stride (giảm giật / giảm tải)
        if (idx % max(1, stride)) != 0:
            if last_result is not None:
                out = draw_yolo_predictions(frame, last_result)
                writer.write(out)
            else:
                writer.write(frame)
            idx += 1
            continue

        frame_bgr = ensure_bgr_ndarray(frame)

        res_list = await asyncio.to_thread(
            model.predict,
            source=frame_bgr,
            imgsz=getattr(settings, "LANE_IMGSZ", 960),
            conf=used_conf,
            iou=0.45,
            max_det=100,
            retina_masks=True,
            save=False,
            verbose=False,
        )
        res = res_list[0]
        last_result = res

        try:
            n_boxes = int(res.boxes.shape[0]) if res.boxes is not None else 0
        except Exception:
            n_boxes = 0

        try:
            n_masks = 0 if (res.masks is None or res.masks.data is None) else int(res.masks.data.shape[0])
        except Exception:
            n_masks = 0

        print(
            f"[lane:video] frame={idx}, boxes={n_boxes}, masks={n_masks}, "
            f"conf={used_conf}, provider={os.environ.get('ORT_PROVIDERS','')}"
        )

        if res.masks is not None and res.masks.data is not None and n_masks > 0:
            print("[lane:video] Drawing masks (res.plot)")
            out = res.plot()
            # hoặc chỉ mask:
            # out = res.plot(labels=False, boxes=False, probs=False)
        else:
            # không in thêm để khỏi spam
            out = frame_bgr

        writer.write(out)
        idx += 1

    cap.release()
    writer.release()
    print(f"[lane:video] DONE. Saved to: {out_path}")
    print(f"[lane:video] ===== END PROCESS VIDEO =====")

    return as_mp4_bytes(out_path)

