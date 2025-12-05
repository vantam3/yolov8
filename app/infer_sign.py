import cv2, os, asyncio
from fastapi import UploadFile
from ultralytics import YOLO
from .models import get_model
from .config import settings
from .utils import read_upload_image, pil_to_bytes, np_bgr_to_pil, save_upload_to_temp_video, ensure_fps, video_writer, as_mp4_bytes, draw_yolo_predictions

# ---------- IMAGE ----------
async def infer_sign_image(image: UploadFile, variant: str, conf: float):
    model: YOLO = get_model("sign", variant)
    img = await read_upload_image(image)
    # Ép task detect
    res = model.predict(source=img, imgsz=settings.SIGN_IMGSZ, conf=conf, task="detect", verbose=False)[0]
    out_bgr = res.plot()  # BGR np.ndarray
    out_pil = np_bgr_to_pil(out_bgr)
    return pil_to_bytes(out_pil, fmt="JPEG", quality=90)

# ---------- VIDEO ----------
async def infer_sign_video(video: UploadFile, variant: str, conf: float, stride: int):
    model: YOLO = get_model("sign", variant)
    src_path = await save_upload_to_temp_video(video)

    cap = cv2.VideoCapture(src_path)
    fps = ensure_fps(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Không đọc được video input.")

    # === ĐỔI Ở ĐÂY: dùng VIDEO_EXT ===
    root, _ = os.path.splitext(src_path)
    out_path = root + ".out" + settings.VIDEO_EXT

    writer = video_writer(out_path, w, h, fps)

    idx = 0
    last_result = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if (idx % stride) != 0:
            # Sticky annotation: vẽ lại kết quả cũ lên frame mới
            if last_result is not None:
                out = draw_yolo_predictions(frame, last_result)
                writer.write(out)
            else:
                writer.write(frame)
            idx += 1
            continue

        # Run async
        res_list = await asyncio.to_thread(
            model.predict,
            source=frame,
            imgsz=settings.SIGN_IMGSZ,
            conf=conf,
            task="detect",
            verbose=False
        )
        last_result = res_list[0]
        
        out = last_result.plot()
        writer.write(out)
        idx += 1

    cap.release()
    writer.release()
    return as_mp4_bytes(out_path)