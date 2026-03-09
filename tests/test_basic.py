import pytest
from video_rag import TextEmbedder, ImageEmbedder, VectorStore, VideoRetriever
from PIL import Image
import numpy as np


def test_text_embedder():
    emb = TextEmbedder()
    v = emb.encode("hello world")
    assert isinstance(v, np.ndarray)
    assert v.shape[-1] > 0


def test_image_embedder():
    emb = ImageEmbedder()
    # create a dummy image
    img = Image.new("RGB", (224, 224), "white")
    v = emb.encode(img)
    assert isinstance(v, np.ndarray)
    assert v.shape[-1] > 0


def test_vector_store():
    store = VectorStore(dim=128)
    emb = np.random.randn(1, 128).astype("float32")
    store.add_text(emb, [{"foo": "bar"}])
    q = emb.copy()
    results = store.search_text(q, k=1)
    assert results
    assert results[0][1]["foo"] == "bar"


def test_retriever_roundtrip():
    retriever = VideoRetriever()
    # simulate simple scene
    scenes = [{"description": "a cat", "frames": []}]
    retriever.index_scenes(scenes, video_name="x")
    res = retriever.query("a cat", k=1)
    assert res and res[0]["description"] == "a cat"
