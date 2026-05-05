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

# Postpone evaluation of type hints. Lets us write `str | Path` and forward
# references like "VideoReader" without runtime cost.
from __future__ import annotations

# Iterator is the abstract type for "thing you can `for ... in`". Used only
# in the type annotation of __iter__ below.
from collections.abc import Iterator
# pathlib.Path: object-oriented filesystem paths, used to make sure the output
# directory exists before we try to write into it.
from pathlib import Path

# OpenCV — the actual video I/O backend.
import cv2
# numpy: cv2 returns frames as numpy arrays of dtype uint8, shape (H, W, 3).
import numpy as np


class VideoReader:
    """Iterable BGR-frame reader with `.fps`, `.size`, `.frame_count`."""

    def __init__(self, path: str | Path):
        # Store path as a string because cv2 expects a string, not a Path object.
        self.path = str(path)
        # Open the video file. cv2.VideoCapture is lazy: it doesn't raise on a
        # bad path — it just returns an object whose .isOpened() reports False.
        self._cap = cv2.VideoCapture(self.path)
        # Convert that silent failure into a real exception so callers can't
        # accidentally process zero frames and not notice.
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")
        # Cache the video's metadata up front so callers can read it as plain
        # attributes (reader.fps, reader.width, ...) without going through
        # cv2.CAP_PROP_* constants every time.
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # NOTE: frame_count is reported by the container metadata and can be
        # slightly off (or zero) for some codecs/streams. It's fine for progress
        # reporting; don't rely on it for correctness.
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
        # __iter__ makes VideoReader directly usable in `for frame in reader:`.
        # We implement it as a generator (`yield` inside) so frames are pulled
        # lazily — only one frame is in memory at a time, regardless of video length.
        try:
            while True:
                # cap.read() returns (ok, frame). `ok` is False when we hit
                # end-of-stream OR a decoder error. Either way we stop.
                ok, frame = self._cap.read()
                if not ok:
                    break
                # `frame` is a numpy array, shape (H, W, 3), dtype uint8, in BGR
                # order (OpenCV's default — NOT RGB).
                yield frame
        finally:
            # Always release the underlying capture, even if the consumer broke
            # out of the loop early or raised an exception. Otherwise the file
            # handle / decoder context can leak until the GC collects it.
            self._cap.release()

    # The two methods below let VideoReader be used as a context manager:
    #   with VideoReader(path) as r: ...
    # which guarantees release() runs even on exceptions inside the `with` block.
    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        # `*_` because we don't care about the (exc_type, exc_val, exc_tb) args
        # — we always release no matter what.
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
        # Same string-coercion trick as VideoReader.
        self.path = str(path)
        # Make sure the output directory exists. Without this, cv2.VideoWriter
        # would silently fail to open if e.g. `results/` didn't exist yet.
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # FOURCC is a 4-character codec identifier. "mp4v" is widely supported
        # and pairs naturally with .mp4 containers. The * unpacks the string
        # into 4 separate char arguments — that's the API VideoWriter_fourcc wants.
        cc = cv2.VideoWriter_fourcc(*fourcc)
        # Construct the writer with the codec, frame rate, and (width, height).
        # Note: cv2 wants size as (width, height), the OPPOSITE of numpy's
        # (height, width) shape — easy to get wrong, hence the `size` tuple.
        self._writer = cv2.VideoWriter(self.path, cc, fps, size)
        # Same silent-failure problem as VideoCapture — convert to an exception.
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        # Forward to the underlying OpenCV writer. `frame` must match the size
        # passed at construction time, otherwise cv2 silently drops it.
        self._writer.write(frame)

    def release(self) -> None:
        # Finalizes the file (writes container trailer, closes handles).
        # MUST be called or the resulting mp4 may be unplayable.
        self._writer.release()

    # Context-manager support — same pattern as VideoReader.
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
