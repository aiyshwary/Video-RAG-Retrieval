# Video RAG Retrieval Project

This repository implements a simple **Video Retrieval-Augmented Generation (RAG)**
system. The goal is to allow semantic search over videos by converting scenes
and frames into embeddings and performing nearest-neighbor lookups.

## Features

- Extract frames and scene descriptions from videos using open-source models
  (``nlpconnect/vit-gpt2-image-captioning`` for captions).
- Compute text embeddings with SentenceTransformers and image embeddings with
  OpenCLIP.
- Store embeddings in FAISS indexes with associated metadata.
- Retrieve relevant scenes or frames for a text query.

> This implementation is a lightweight Python prototype and uses FAISS for the
> vector store instead of a full VDMS server mentioned in the original design.

## Installation

```bash
git clone <repo-url>
cd video-rag-retrieval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` includes all necessary Python packages. A GPU is
optional but speeds up embedding computation.

## Example usage

```python
from video_rag import extract_scenes, VideoRetriever

# process a video and index its scenes
scenes = extract_scenes("path/to/video.mp4", fps=0.5)
retriever = VideoRetriever()
retriever.index_scenes(scenes, video_name="demo")

# query
results = retriever.query("a person walking", k=3, use_text=True)
for r in results:
    print(r)
```

## Project layout

```
/Video RAG Retrieval Project
├── README.md
├── requirements.txt
└── src/video_rag/
    ├── __init__.py
    ├── preprocessing.py
    ├── embeddings.py
    ├── storage.py
    ├── retrieval.py
    └── utils.py
```

## Notes

- Scene detection is rudimentary; you can replace `extract_scenes` with more
  advanced shot-detection or segmentation logic.
- Metadata stored with embeddings can be extended to track file paths,
  timestamps, etc.
- For a production system, swap the FAISS store for a proper vector database
  such as VDMS, Milvus, or Pinecone.

---

Feel free to extend this prototype with additional functionality.