"""Embedding utilities for text and images."""

from typing import List, Union
import numpy as np

from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import open_clip


class TextEmbedder:
    """Wrapper around a sentence-transformers model for text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)


class ImageEmbedder:
    """Wrapper around an OpenCLIP model for image embeddings."""

    def __init__(self, model_name: str = "ViT-g-14", pretrained: str = "laion2b_s12b_b42k"):
        """Load an OpenCLIP model.

        The default tag ``laion2b_s12b_b42k`` is available in the version of
        ``open_clip_torch`` installed in this environment.  You can override the
        tag if you have a different checkpoint locally or in the HF hub.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except RuntimeError as e:
            # fallback to unpretrained if tag isn't recognized
            print(f"warning: failed to load pretrained clip weights ({e}), using random init")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=None
            )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """Return numpy array with shape (n, dim) for a list of PIL images."""
        single = False
        if isinstance(images, Image.Image):
            images = [images]
            single = True
        tensors = []
        for img in images:
            tensors.append(self.preprocess(img).unsqueeze(0))
        batch = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
        feats = feats.cpu().numpy()
        # normalize to unit length
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(norms, 1e-12)
        return feats if not single else feats[0]
