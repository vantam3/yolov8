import io
import os
import tempfile

import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile

from .config import settings


# =========================
# ẢNH
# =========================

def pil_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> io.BytesIO:
    """
    Convert PIL Image -> BytesIO để trả về response.
    """
    buf = io.BytesIO()
    img.save(buf, fmt, quality=quality)
    buf.seek(0)
    return buf


def np_bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """
    OpenCV BGR ndarray -> PIL RGB Image.
    """
    return Image.fromarray(bgr[:, :, ::-1])


async def read_upload_image(file: UploadFile) -> Image.Image:
    """
    Đọc file upload từ FastAPI thành PIL Image (RGB).
    """
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return img


# =========================
# VIDEO
# =========================

def ensure_fps(cap: cv2.VideoCapture) -> float:
    """
    Lấy FPS từ video, nếu không hợp lệ thì dùng fallback từ config.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # fps != fps là check NaN
    if fps <= 0 or fps != fps:
        fps = settings.VIDEO_FPS_FALLBACK
    return float(fps)


async def save_upload_to_temp_video(file: UploadFile) -> str:
    """
    Lưu UploadFile (video) ra 1 file tạm và trả về path.
    """
    # Lấy đuôi file gốc, nếu không có thì mặc định .mp4
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        data = await file.read()
        f.write(data)
        return f.name


def video_writer(path_out: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """
    Tạo VideoWriter.

    - Nếu ext là .mp4/.mov/.m4v -> dùng codec 'mp4v' (ổn trên Windows, container MP4).
    - Nếu ext là .webm -> ưu tiên settings.VIDEO_FOURCC (thường là 'VP80'), fallback 'VP80'.
    - Ngược lại -> dùng settings.VIDEO_FOURCC nếu có, không thì 'mp4v'.
    """
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height không hợp lệ: {width}x{height}")

    fourcc_func = getattr(cv2, "VideoWriter_fourcc", None)
    if fourcc_func is None:
        raise RuntimeError("OpenCV không có hàm VideoWriter_fourcc – kiểm tra lại cài đặt cv2.")

    ext = os.path.splitext(path_out)[1].lower()

    if ext in [".mp4", ".m4v", ".mov"]:
        fourcc_str = "mp4v"
    elif ext in [".webm"]:
        # ưu tiên config, nếu không có thì dùng VP80 cho webm
        cfg = getattr(settings, "VIDEO_FOURCC", None)
        fourcc_str = cfg if (isinstance(cfg, str) and len(cfg) == 4) else "VP80"
    else:
        # fallback chung
        cfg = getattr(settings, "VIDEO_FOURCC", None)
        fourcc_str = cfg if (isinstance(cfg, str) and len(cfg) == 4) else "mp4v"

    fourcc = fourcc_func(*fourcc_str)
    writer = cv2.VideoWriter(path_out, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(
            f"Không tạo được VideoWriter cho '{path_out}' với codec '{fourcc_str}'. "
            f"(ext={ext}, w={width}, h={height}, fps={fps})"
        )

    print(f"[video_writer] Opened writer: path={path_out}, codec={fourcc_str}, size={width}x{height}, fps={fps}")
    return writer


def as_mp4_bytes(path_out: str) -> io.BytesIO:
    """
    Đọc file video (dù ext là .webm hay .mp4) -> BytesIO để StreamingResponse trả về.
    Tên hàm cũ 'as_mp4_bytes' nhưng dùng được cho mọi loại video.
    """
    with open(path_out, "rb") as f:
        data = f.read()
    buf = io.BytesIO(data)
    buf.seek(0)
    return buf


def ensure_names_and_plot(res):
    """
    Vá thiếu names khi plot kết quả Ultralytics:
    - Nếu có detection nhưng res.names thiếu key cho class-id -> tự tạo names = {0: 'class0', ...}
    - Sau đó gọi res.plot() như bình thường.
    """
    max_cls = -1
    try:
        if getattr(res, "boxes", None) is not None and res.boxes.cls is not None and len(res.boxes) > 0:
            max_cls = int(res.boxes.cls.max().item())
    except Exception:
        pass

    need_fix = False
    names = getattr(res, "names", None)

    if isinstance(names, dict):
        if max_cls >= 0 and max_cls not in names:
            need_fix = True
    elif isinstance(names, list):
        if max_cls >= 0 and max_cls >= len(names):
            need_fix = True
    else:
        # names None hoặc kiểu lạ -> cần fix nếu có detection
        if max_cls >= 0:
            need_fix = True

    if need_fix:
        # tạo mapping tối thiểu để không văng lỗi khi plot
        n = max(max_cls + 1, 1)
        res.names = {i: f"class{i}" for i in range(n)}

    return res.plot()


def draw_yolo_predictions(img: np.ndarray, result) -> np.ndarray:
    """
    Vẽ lại kết quả detection/segmentation (result) lên một ảnh mới (img).
    Dùng để 'giữ' annotation trên các frame bị skip (sticky annotations).
    """
    if result is None:
        return img

    # result.plot(img=...) sẽ vẽ boxes/masks/labels lên img được truyền vào
    # Lưu ý: result.plot() trả về ảnh BGR mới (copy), không sửa in-place ảnh gốc
    return result.plot(img=img)
