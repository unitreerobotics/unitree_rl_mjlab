"""Depth frames for the mapper. The simulator's comes from shared memory (simulate/src/depth_camera.h);
the R1's camera module gets its own source with the same read() once its output is known."""

from __future__ import annotations

import mmap
import os
import struct
import time

import numpy as np

from .mapper import Intrinsics

# simulate/src/depth_camera.h DepthShmHeader: magic, header_size, width, height, fx, fy, cx, cy,
# depth_unit, reserved, seq (at byte 48), stamp, sim_time.
_HEADER = struct.Struct("<8sIIIfffffI4xQdd")
_SEQ = 48


class ShmDepthSource:
    def __init__(self, name: str = "/r1_depth", max_age_s: float = 1.0):
        self.path = "/dev/shm/" + name.lstrip("/")
        self.max_age_s = max_age_s
        self._mm: mmap.mmap | None = None
        self._inode: int | None = None
        self._seq = 0
        self.skipped = 0   # frames the simulator wrote that were overwritten before read() saw them

    def _open(self) -> bool:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            self._mm = None
            return False
        if self._mm is not None and st.st_ino == self._inode:
            return True
        # (Re)opened: the simulator restarted and made a new file.
        with open(self.path, "rb") as f:
            self._mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        self._inode, self._seq = st.st_ino, 0
        return True

    def read(self):
        """(depth in metres with NaN where none, Intrinsics, capture wall time) for a frame not read
        before and at most max_age_s old; else None."""
        if not self._open() or len(self._mm) < _HEADER.size:
            return None
        for _ in range(3):
            seq = struct.unpack_from("<Q", self._mm, _SEQ)[0]
            if seq % 2 or seq == self._seq:
                return None
            h = _HEADER.unpack_from(self._mm, 0)
            if h[0] != b"R1DEPTH1":
                return None
            raw = np.frombuffer(self._mm, np.uint16, h[2] * h[3], h[1]).reshape(h[3], h[2]).copy()
            if struct.unpack_from("<Q", self._mm, _SEQ)[0] == seq:
                break
        else:
            return None
        if self._seq and seq > self._seq + 2:
            self.skipped += (seq - self._seq) // 2 - 1   # seqlock: +2 per frame
        self._seq = seq
        stamp = h[11]
        if time.time() - stamp > self.max_age_s:
            return None  # left over from a simulator that is gone
        depth = raw.astype(np.float32) * h[8]
        depth[raw == 0] = np.nan
        return depth, Intrinsics(h[4], h[5], h[6], h[7]), stamp
