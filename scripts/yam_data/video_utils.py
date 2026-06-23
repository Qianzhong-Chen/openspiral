"""
Video processing utilities for YAMS data conversion.

This module contains optimized video encoding/decoding functions with hardware acceleration support.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import warnings

import cv2
import ffmpeg
import numpy as np

warnings.filterwarnings("ignore")

# Try to import faster libraries
try:
    import av  # PyAV for faster video decoding

    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

try:
    import torch
    import torchvision.transforms.functional as TF

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def detect_hardware_encoder() -> list[str]:
    """Detect available hardware video encoders."""
    encoders = []

    # Test for NVIDIA NVENC
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10, check=False
        )
        if "h264_nvenc" in result.stdout:
            encoders.append("h264_nvenc")
        if "hevc_nvenc" in result.stdout:
            encoders.append("hevc_nvenc")
    except:
        pass

    # Test for Intel Quick Sync
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10, check=False
        )
        if "h264_qsv" in result.stdout:
            encoders.append("h264_qsv")
    except:
        pass

    # Test for AMD VCE
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10, check=False
        )
        if "h264_amf" in result.stdout:
            encoders.append("h264_amf")
    except:
        pass

    # Fallback to fast software encoders
    encoders.extend(["libx264", "libx265"])

    return encoders


def encode_video_hardware(frames: list[np.ndarray], save_path: Path, fps: int = 30, encoder: str = None):
    """Encode frames using hardware acceleration when available."""
    if not frames:
        return

    if encoder is None:
        available_encoders = detect_hardware_encoder()
        encoder = available_encoders[0] if available_encoders else "libx264"

    height, width, _ = frames[0].shape

    # Create temporary file for input
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as temp_file:
        temp_path = temp_file.name
        # Write frames as raw video data
        frame_data = np.stack(frames).astype(np.uint8)
        temp_file.write(frame_data.tobytes())

    try:
        # Configure encoder-specific settings
        if "nvenc" in encoder:
            # NVIDIA NVENC settings for maximum speed
            encoder_args = {
                "vcodec": encoder,
                "preset": "p1",  # Fastest preset for NVENC
                "tune": "ull",  # Ultra-low latency
                "rc": "vbr",  # Variable bitrate
                "cq": "23",  # Quality level
                "b:v": "5M",  # Target bitrate
                "maxrate": "10M",
                "bufsize": "10M",
                "gpu": "0",  # Use first GPU
            }
        elif "qsv" in encoder:
            # Intel Quick Sync settings
            encoder_args = {"vcodec": encoder, "preset": "veryfast", "global_quality": "23", "look_ahead": "0"}
        elif "amf" in encoder:
            # AMD VCE settings
            encoder_args = {"vcodec": encoder, "quality": "speed", "rc": "vbr_peak", "qp_i": "22", "qp_p": "24"}
        else:
            # Fast software encoding fallback
            encoder_args = {"vcodec": encoder, "preset": "ultrafast", "crf": "23", "tune": "fastdecode"}

        # Common settings
        encoder_args.update(
            {
                "pix_fmt": "yuv420p",
                "r": fps,
                "movflags": "+faststart",  # Enable fast start for web playback
            }
        )

        # Build ffmpeg command
        input_stream = ffmpeg.input(temp_path, format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}", framerate=fps)

        output_stream = ffmpeg.output(input_stream, str(save_path), **encoder_args)

        # Run with minimal overhead
        ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True, quiet=True)

    except Exception as e:
        print(f"Hardware encoding failed with {encoder}, falling back to libx264: {e}")
        # Fallback to basic libx264
        try:
            (
                ffmpeg.input(temp_path, format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}", framerate=fps)
                .output(str(save_path), vcodec="libx264", preset="ultrafast", crf=23, pix_fmt="yuv420p", r=fps)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
        except Exception:
            pass  # Silent fallback failure
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def extract_video_frames_fast(video_path: Path) -> np.ndarray:
    """Extract frames from video using the fastest available method."""
    if HAS_PYAV:
        return extract_video_frames_pyav(video_path)
    return extract_video_frames_opencv_optimized(video_path)


def extract_video_frames_pyav(video_path: Path) -> np.ndarray:
    """Extract frames using PyAV (much faster than OpenCV)."""
    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]

        # Pre-allocate array
        frame_count = video_stream.frames
        if frame_count == 0:  # Fallback if frame count unknown
            frame_count = int(video_stream.duration * video_stream.average_rate)

        width = video_stream.width
        height = video_stream.height
        frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)

        frame_idx = 0
        for frame in container.decode(video_stream):
            if frame_idx >= frame_count:
                break
            # Convert to RGB numpy array
            frames[frame_idx] = frame.to_ndarray(format="rgb24")
            frame_idx += 1

        container.close()
        return frames[:frame_idx]

    except Exception:
        # Fallback to OpenCV silently
        return extract_video_frames_opencv_optimized(video_path)


def extract_video_frames_opencv_optimized(video_path: Path) -> np.ndarray:
    """Optimized OpenCV video extraction with better buffering."""
    cap = cv2.VideoCapture(str(video_path))

    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Get properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-allocate with exact size
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)

    # Read all frames in batch
    frame_idx = 0
    while frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        # Direct BGR to RGB conversion
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=frames[frame_idx])
        frame_idx += 1

    cap.release()
    return frames[:frame_idx]


def resize_frames_vectorized(frames: np.ndarray, target_size: int) -> np.ndarray:
    """Vectorized frame resizing using the fastest available method."""
    if HAS_TORCH:
        return resize_frames_torch(frames, target_size)
    return resize_frames_numpy_vectorized(frames, target_size)


def resize_frames_torch(frames: np.ndarray, target_size: int) -> np.ndarray:
    """Ultra-fast GPU/CPU vectorized resizing using PyTorch."""
    # Convert to torch tensor (HWC -> CHW)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    # Resize all frames at once
    resized = TF.resize(tensor, [target_size, target_size], antialias=True)

    # Convert back to numpy (CHW -> HWC)
    result = (resized.permute(0, 2, 3, 1) * 255).byte().numpy()
    return result


def resize_frames_numpy_vectorized(frames: np.ndarray, target_size: int) -> np.ndarray:
    """Vectorized numpy-based resizing (faster than PIL loop)."""
    n_frames, h, w, c = frames.shape

    # Calculate resize parameters
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize all frames using OpenCV (faster than PIL for batch)
    resized_frames = np.empty((n_frames, target_size, target_size, c), dtype=np.uint8)

    for i in range(n_frames):
        # Resize frame
        resized = cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Add padding
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2

        resized_frames[i] = np.pad(
            resized,
            ((pad_h, target_size - new_h - pad_h), (pad_w, target_size - new_w - pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return resized_frames
