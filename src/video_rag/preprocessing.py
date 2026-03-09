"""Video preprocessing utilities.

Functions to extract frames and generate simple scene descriptions.
"""

import os
from typing import List, Dict, Any
import cv2
from PIL import Image

from transformers import pipeline


# create a global captioning pipeline to avoid reloading for each frame
_captioner = None

def _get_captioner():
    global _captioner
    if _captioner is None:
        # note: this will download the model on first use
        _captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return _captioner


def sample_frames(video_path: str, fps: float = 1.0) -> List[Dict[str, Any]]:
    """Sample frames from a video at roughly ``fps`` frames per second.

    Returns a list of dictionaries with keys:
      - ``frame``: PIL.Image
      - ``timestamp``: float (seconds)
      - ``frame_index``: int
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(video_fps / fps)
    frames = []
    idx = 0
    grabbed = True
    while grabbed:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if idx % step == 0:
            # convert BGR to RGB and to PIL
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img)
            timestamp = idx / video_fps
            frames.append(
                {"frame": pil, "timestamp": timestamp, "frame_index": idx}
            )
        idx += 1
    cap.release()
    return frames


def extract_scenes(video_path: str, fps: float = 1.0) -> List[Dict[str, Any]]:
    """Simplistic "scene" extraction by captioning sampled frames.

    This function samples frames (``sample_frames``) and generates a
    text caption for each frame. It then groups consecutive frames with the
    same caption into a single scene description. The grouping logic is
    intentionally very simple; a more advanced shot-detection model can be
    substituted if desired.
    """
    frames = sample_frames(video_path, fps=fps)
    captioner = _get_captioner()
    scenes: List[Dict[str, Any]] = []
    for item in frames:
        text = captioner(item["frame"])[0]["generated_text"]
        if scenes and scenes[-1]["description"] == text:
            scenes[-1]["end_time"] = item["timestamp"]
            scenes[-1]["frames"].append(item)
        else:
            scenes.append(
                {
                    "description": text,
                    "start_time": item["timestamp"],
                    "end_time": item["timestamp"],
                    "frames": [item],
                }
            )
    return scenes
