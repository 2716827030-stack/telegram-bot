"""隐写术检测与解码模块 - 使用官方 DuckDuckGoose 算法

基于 https://github.com/copyangle/SS_tools 的官方算法
"""

import io
import os
import struct
import tempfile
import logging
from typing import Optional, Tuple, List
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

WATERMARK_SKIP_W_RATIO = 0.40
WATERMARK_SKIP_H_RATIO = 0.08


def _extract_payload_with_k(arr: np.ndarray, k: int) -> bytes:
    h, w, c = arr.shape
    skip_w = int(w * WATERMARK_SKIP_W_RATIO)
    skip_h = int(h * WATERMARK_SKIP_H_RATIO)
    mask2d = np.ones((h, w), dtype=bool)
    if skip_w > 0 and skip_h > 0:
        mask2d[:skip_h, :skip_w] = False
    mask3d = np.repeat(mask2d[:, :, None], c, axis=2)
    flat = arr.reshape(-1)
    idxs = np.flatnonzero(mask3d.reshape(-1))
    vals = (flat[idxs] & ((1 << k) - 1)).astype(np.uint8)
    ub = np.unpackbits(vals, bitorder="big").reshape(-1, 8)[:, -k:]
    bits = ub.reshape(-1)
    if len(bits) < 32:
        raise ValueError("Insufficient image data. 图像数据不足")
    len_bits = bits[:32]
    length_bytes = np.packbits(len_bits, bitorder="big").tobytes()
    header_len = struct.unpack(">I", length_bytes)[0]
    total_bits = 32 + header_len * 8
    if header_len <= 0 or total_bits > len(bits):
        raise ValueError("Payload length invalid. 载荷长度异常")
    payload_bits = bits[32:32 + header_len * 8]
    return np.packbits(payload_bits, bitorder="big").tobytes()


def _parse_header(header: bytes, password: str = ""):
    idx = 0
    if len(header) < 1:
        raise ValueError("Header corrupted. 文件头损坏")
    has_pwd = header[0] == 1
    idx += 1
    pwd_hash = b""
    salt = b""
    if has_pwd:
        if len(header) < idx + 32 + 16:
            raise ValueError("Header corrupted. 文件头损坏")
        pwd_hash = header[idx:idx + 32]
        idx += 32
        salt = header[idx:idx + 16]
        idx += 16
    if len(header) < idx + 1:
        raise ValueError("Header corrupted. 文件头损坏")
    ext_len = header[idx]
    idx += 1
    if len(header) < idx + ext_len + 4:
        raise ValueError("Header corrupted. 文件头损坏")
    ext = header[idx:idx + ext_len].decode("utf-8", errors="ignore")
    idx += ext_len
    data_len = struct.unpack(">I", header[idx:idx + 4])[0]
    idx += 4
    data = header[idx:]
    if len(data) != data_len:
        raise ValueError("Data length mismatch. 数据长度不匹配")
    if not has_pwd:
        return data, ext
    if not password:
        raise ValueError("Password required. 需要密码")
    import hashlib
    check_hash = hashlib.sha256((password + salt.hex()).encode("utf-8")).digest()
    if check_hash != pwd_hash:
        raise ValueError("Wrong password. 密码错误")
    key_material = (password + salt.hex()).encode("utf-8")
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        out.extend(hashlib.sha256(key_material + str(counter).encode("utf-8")).digest())
        counter += 1
    ks = bytes(out[:len(data)])
    plain = bytes(a ^ b for a, b in zip(data, ks))
    return plain, ext


def binpng_bytes_to_mp4_bytes(raw_data: bytes) -> bytes:
    """将binpng数据转换为mp4数据"""
    img = Image.open(io.BytesIO(raw_data)).convert("RGB")
    arr = np.array(img).astype(np.uint8)
    flat = arr.reshape(-1, 3).reshape(-1)
    return flat.tobytes().rstrip(b"\x00")


def detect_and_decode(image_data: bytes, password: str = "") -> Tuple[bool, Optional[str], Optional[bytes], Optional[str]]:
    """检测并解码图像中的隐写信息
    
    Returns: (成功, 文本, 原始数据, 扩展名)
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        arr = np.array(image.convert("RGB")).astype(np.uint8)
        logger.info(f"开始检测... 图像尺寸: {image.size}")
        
        header = None
        raw = None
        ext = None
        last_err = None
        
        # 尝试 k=2,6,8
        for k in (2, 6, 8):
            try:
                logger.debug(f"尝试模式 k={k}")
                header = _extract_payload_with_k(arr, k)
                raw, ext = _parse_header(header, password)
                logger.info(f"成功！模式 k={k}, 扩展名: {ext}")
                break
            except Exception as e:
                logger.debug(f"模式 k={k} 失败: {e}")
                last_err = e
                continue
        
        if raw is None:
            logger.info("未检测到有效隐写内容")
            return False, None, None, None
        
        # 处理 binpng 格式！
        if ext.endswith(".binpng"):
            logger.info(f"检测到 binpng 格式，正在解码为 mp4")
            mp4_bytes = binpng_bytes_to_mp4_bytes(raw)
            raw = mp4_bytes
            ext = "mp4"
            logger.info(f"转换成功，得到 mp4 数据")
        
        # 处理文本
        text_output = ""
        if ext.lower() == "txt":
            try:
                text_output = raw.decode("utf-8")
            except Exception:
                try:
                    text_output = raw.decode("gbk")
                except Exception:
                    text_output = f"解码文本失败 (扩展名: {ext})"
        
        return True, text_output, raw, ext
        
    except Exception as e:
        logger.error(f"解码过程异常: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False, None, None, None


def extract_video_frames(video_bytes: bytes) -> List[bytes]:
    """从视频字节数据中提取所有帧作为图片字节
    
    Returns: 每一帧的PNG图片字节列表
    """
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name
            f.write(video_bytes)
        
        # 使用 imageio 读取视频
        import imageio.v3 as iio
        frames = []
        for idx, frame in enumerate(iio.imiter(temp_path, plugin="pyav")):
            # 转换为 PIL 图片并保存为 PNG
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            frames.append(buf.getvalue())
            logger.info(f"提取到第 {idx+1} 帧")
        
        # 删除临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return frames
    except Exception as e:
        logger.error(f"提取视频帧失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []


def extract_video_frames_opencv(video_bytes: bytes) -> List[bytes]:
    """备用方案：使用 OpenCV 提取视频帧"""
    try:
        import cv2
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name
            f.write(video_bytes)
        
        cap = cv2.VideoCapture(temp_path)
        frames = []
        idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            frames.append(buf.getvalue())
            logger.info(f"提取到第 {idx+1} 帧")
            idx += 1
        
        cap.release()
        
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return frames
    except Exception as e:
        logger.error(f"OpenCV 提取视频帧失败: {e}")
        return []

