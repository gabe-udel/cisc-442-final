# video_io.py - thin wrappers around cv2 for reading and writing video files

from pathlib import Path

import cv2


class VideoReader:

    def __init__(self, path):
        # cv2 needs a string, not a Path object
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self):
        # (width, height) in the order cv2.VideoWriter expects
        return (self.width, self.height)

    def __iter__(self):
        # yield frames one at a time so only one is in memory at once
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                yield frame
        finally:
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._cap.release()


class VideoWriter:

    def __init__(self, path, fps, size, fourcc="mp4v"):
        self.path = str(path)
        # make sure the output folder exists before trying to write
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        cc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer = cv2.VideoWriter(self.path, cc, fps, size)
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for: {self.path}")

    def write(self, frame):
        self._writer.write(frame)

    def release(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


# open a video file for reading
def read_video(path):
    return VideoReader(path)


# create a writer that matches the reader's fps and size
def writer_matching(path, reader, fourcc="mp4v"):
    return VideoWriter(path, fps=reader.fps, size=reader.size, fourcc=fourcc)
