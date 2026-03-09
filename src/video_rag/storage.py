"""Vector storage using FAISS for quick similarity search.

This is a simplified substitute for a proper vector database (VDMS) described
in the project specification. The store keeps two separate FAISS indexes (text
and image) plus Python-side metadata, and exposes simple add/search APIs.
"""

from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        # one index for text, one for image
        self.text_index = faiss.IndexFlatIP(dim)
        self.image_index = faiss.IndexFlatIP(dim)
        self.text_metadata: List[Dict[str, Any]] = []
        self.image_metadata: List[Dict[str, Any]] = []

    def add_text(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Add text embeddings with corresponding metadata."""
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        faiss.normalize_L2(embeddings)
        self.text_index.add(embeddings.astype('float32'))
        self.text_metadata.extend(metadatas)

    def add_images(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Add image embeddings with metadata."""
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        faiss.normalize_L2(embeddings)
        self.image_index.add(embeddings.astype('float32'))
        self.image_metadata.extend(metadatas)

    def search_text(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Return top-k nearest text results with similarity scores."""
        if query_emb.ndim == 1:
            query_emb = query_emb[np.newaxis, :]
        faiss.normalize_L2(query_emb)
        D, I = self.text_index.search(query_emb.astype('float32'), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((float(dist), self.text_metadata[idx]))
        return results

    def search_image(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Return top-k nearest image results with similarity scores."""
        if query_emb.ndim == 1:
            query_emb = query_emb[np.newaxis, :]
        faiss.normalize_L2(query_emb)
        D, I = self.image_index.search(query_emb.astype('float32'), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((float(dist), self.image_metadata[idx]))
        return results
