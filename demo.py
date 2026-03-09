"""
demo.py – end-to-end demonstration of the Video RAG Retrieval pipeline.

Because we may not have a real video file handy, this script:
  1. Synthesises a short test video (30 frames, 5 fps) using OpenCV.
  2. Indexes it with VideoRetriever (text embeddings only, no heavy vision model).
  3. Runs several text queries and prints the ranked results.
"""

import os, sys, textwrap
import cv2
import numpy as np

# ── 1. Synthesise a short test video ─────────────────────────────────────────
VIDEO_PATH = "/tmp/video_rag_demo.mp4"
W, H, FPS, TOTAL = 640, 360, 5, 30  # 6-second clip

print("=" * 64)
print("  Video RAG Retrieval Project  –  Demo")
print("=" * 64)
print(f"\n[1/4] Generating synthetic test video → {VIDEO_PATH}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (W, H))

# draw 6 distinct "scenes" (5 frames each)
scene_defs = [
    ((30,  40,  60),  "walking",    "person walking on street"),
    ((20,  50,  20),  "park",       "people relaxing in park"),
    ((60,  30,  20),  "store",      "person picking item in store"),
    ((50,  50,  80),  "office",     "meeting in conference room"),
    ((80,  60,  20),  "road",       "car driving on highway"),
    ((20,  30,  70),  "night",      "city at night, bright lights"),
]

for scene_idx, (bg_color, label, _) in enumerate(scene_defs):
    for _ in range(5):
        frame = np.full((H, W, 3), bg_color, dtype=np.uint8)
        cv2.putText(frame, f"Scene {scene_idx + 1}: {label}",
                    (30, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (220, 220, 220), 2, cv2.LINE_AA)
        writer.write(frame)
writer.release()
print(f"   → {TOTAL} frames written  ({len(scene_defs)} scenes × 5 frames @ {FPS} fps)")

# ── 2. Manually build scene descriptors (skip heavy vision model) ────────────
print("\n[2/4] Building scene descriptors (using ground-truth captions for demo)")

scenes = []
for i, (_, label, description) in enumerate(scene_defs):
    start_t = i * (5 / FPS)
    end_t   = start_t + (5 / FPS) - 0.01
    scenes.append({
        "description": description,
        "start_time":  round(start_t, 2),
        "end_time":    round(end_t,   2),
        "frames": [],          # no actual PIL frames needed for text-only demo
    })
    print(f'   Scene {i+1}: [{start_t:.1f}s - {end_t:.1f}s]  "{description}"')

# ── 3. Index scenes into the retriever ───────────────────────────────────────
print("\n[3/4] Indexing scenes into VideoRetriever (text embeddings) …")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from video_rag.embeddings import TextEmbedder
from video_rag.storage    import VectorStore

embedder = TextEmbedder()            # all-MiniLM-L6-v2
sample   = embedder.encode("")
store    = VectorStore(dim=sample.shape[-1])

texts    = [s["description"] for s in scenes]
embs     = embedder.encode(texts)
metas    = [
    {
        "video":       "demo.mp4",
        "scene_index": i,
        "description": s["description"],
        "start_time":  s["start_time"],
        "end_time":    s["end_time"],
    }
    for i, s in enumerate(scenes)
]
store.add_text(embs, metas)
print(f"   → {len(scenes)} scenes indexed  (embedding dim = {sample.shape[-1]})")

# ── 4. Run queries ────────────────────────────────────────────────────────────
queries = [
    "a person walking",
    "car on a road",
    "night city lights",
    "indoor meeting with colleagues",
    "someone shopping",
]

print("\n[4/4] Running retrieval queries\n")
print("-" * 64)

for q in queries:
    q_emb    = embedder.encode(q)
    results  = store.search_text(q_emb, k=3)
    top_sim, top_meta = results[0]
    print(f'  Query : "{q}"')
    print(f"  Result: Scene {top_meta['scene_index']+1}  |  "
          f"{top_meta['start_time']}s - {top_meta['end_time']}s  |  "
          f"sim={top_sim:.3f}")
    desc = top_meta['description']
    print(f'          "{desc}"')
    print()

print("-" * 64)
print("\n✓ Demo complete.")
