"""High-level retrieval interface combining the components."""

from typing import Optional, List, Dict, Any

import numpy as np
from .embeddings import TextEmbedder, ImageEmbedder
from .storage import VectorStore


class VideoRetriever:
    def __init__(
        self,
        text_model: Optional[str] = None,
        image_model: Optional[str] = None,
    ):
        """Create a retriever with its own embedders and vector store.

        The store dimension is inferred from the text embedder output.  Both the
        text and image embedders are expected to produce vectors of the same
        size; if this is not true you can wrap them yourself and create a
        custom ``VectorStore`` instance.
        """
        self.text_embedder = TextEmbedder(model_name=text_model) if text_model else TextEmbedder()
        self.image_embedder = ImageEmbedder(model_name=image_model) if image_model else ImageEmbedder()
        # infer dim by encoding an empty string
        sample = self.text_embedder.encode("")
        dim = sample.shape[-1]
        self.store = VectorStore(dim)

    def index_scenes(self, scenes: List[Dict[str, Any]], video_name: str):
        """Index the list of scenes returned by ``extract_scenes``.

        Each scene should contain a ``description`` string and a list of
        ``frames`` (each with a ``frame`` PIL image, ``timestamp`` and ``frame_index``).
        """
        texts = [s["description"] for s in scenes]
        text_emb = self.text_embedder.encode(texts)
        text_metas = []
        image_embs = []
        image_metas = []
        for s_idx, scene in enumerate(scenes):
            for frame_obj in scene.get("frames", []):
                image_embs.append(self.image_embedder.encode(frame_obj["frame"]))
                image_metas.append(
                    {
                        "video": video_name,
                        "scene_index": s_idx,
                        "timestamp": frame_obj["timestamp"],
                        "frame_index": frame_obj["frame_index"],
                        "scene_description": scene["description"],
                    }
                )
            text_metas.append({"video": video_name, "scene_index": s_idx, "description": scene["description"]})
        if len(text_emb) > 0:
            self.store.add_text(text_emb, text_metas)
        if len(image_embs) > 0:
            self.store.add_images(np.vstack(image_embs), image_metas)

    def query(self, prompt: str, k: int = 5, use_text: bool = True, use_image: bool = False):
        """Run a retrieval for the given text prompt.

        ``use_text``/``use_image`` flags control which indexes to query. Results
        from both sources are concatenated but no further re-ranking is performed.
        """
        results: List[Dict[str, Any]] = []
        if use_text:
            q_emb = self.text_embedder.encode(prompt)
            text_res = self.store.search_text(q_emb, k=k)
            for sim, meta in text_res:
                results.append({"score": sim, "type": "text", **meta})
        if use_image:
            q_emb = self.image_embedder.encode(prompt)
            img_res = self.store.search_image(q_emb, k=k)
            for sim, meta in img_res:
                results.append({"score": sim, "type": "image", **meta})
        # sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
