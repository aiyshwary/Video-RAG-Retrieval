from setuptools import setup, find_packages

setup(
    name="video_rag",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "sentence-transformers",
        "open_clip_torch",
        "opencv-python",
        "ffmpeg-python",
        "faiss-cpu",
    ],
    description="Video retrieval-augmented generation prototype",
    author="",
)
