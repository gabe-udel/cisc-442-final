"""Thin wrappers around cv2.VideoCapture / cv2.VideoWriter.

Designed for the frame-by-frame pose pipeline: open a reader, iterate frames
(BGR numpy arrays), pipe each one through any per-frame processing, and feed
the result into a writer that matches the reader's fps and size.

Why wrap OpenCV at all? Because cv2.VideoCapture/VideoWriter have an awkward,
C-style API: you have to remember to call .release(), check .isOpened(), poll
for properties, and read in a `while True` loop. The classes below put a
Pythonic skin on that — context-manager support, iteration, and named
attributes — so the calling code (main.py) reads cleanly.
"""


from __future__ import annotations


from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Iterable BGR-frame reader with `.fps`, `.size`, `.frame_count`."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) — matches the order cv2.VideoWriter expects.

        Exposed as a property (not stored) so it always reflects width/height
        even if those got updated. Also lets us pass `reader.size` straight to
        the VideoWriter constructor with no fuss.
        """
        return (self.width, self.height)

    def __iter__(self) -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                yield frame
        finally:
            self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self._cap.release()


class VideoWriter:
    """BGR-frame writer. Use as a context manager or call `.release()` yourself."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        fourcc: str = "mp4v",
    ):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        cc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer = cv2.VideoWriter(self.path, cc, fps, size)
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()
    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()


def read_video(path: str | Path) -> VideoReader:
    """Open `path` for reading. Iterate the returned object to get BGR frames.

    Tiny convenience wrapper so callers don't have to import the class; reads
    nicely as `reader = read_video("clip.mp4")`.
    """
    return VideoReader(path)


def writer_matching(path: str | Path, reader: VideoReader, fourcc: str = "mp4v") -> VideoWriter:
    """Convenience: create a VideoWriter with the same fps and size as `reader`.

    Almost every output video in this project should match its input's fps and
    resolution — mismatches cause warped or fast-forwarded playback. This helper
    means you never have to type `fps=reader.fps, size=reader.size` by hand.
    """
    return VideoWriter(path, fps=reader.fps, size=reader.size, fourcc=fourcc)
