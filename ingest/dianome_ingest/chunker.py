"""Physical layer: fixed 8 MB content-addressed chunks.

The chunker knows nothing about tensors. It takes byte *streams* and cuts each
one independently into 8,388,608-byte chunks (the last chunk of a stream may be
shorter). Chunk id = lowercase hex SHA-256 of the chunk bytes.

Chunk boundaries restart at every stream. That is what makes the store dedup:
two streams with identical bytes (for example the q8 embed group in the q8
and q4 variants) cut into identical chunks, wherever they sit in their model.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Iterable

CHUNK_SIZE = 8 * 1024 * 1024  # 8,388,608

Sink = Callable[[str, bytes], None]
ChunkList = list[tuple[str, int]]


class StreamChunker:
    """Incremental chunker for one stream. write() pieces, then finish()."""

    def __init__(self, sink: Sink, seen: set[str] | None = None):
        self._sink = sink
        self._seen = seen if seen is not None else set()
        self._buf = bytearray()
        self._out: ChunkList = []
        self.length = 0

    def write(self, data: bytes | bytearray | memoryview) -> None:
        mv = memoryview(data).cast("B")
        n = len(mv)
        pos = 0
        while pos < n:
            take = min(CHUNK_SIZE - len(self._buf), n - pos)
            self._buf += mv[pos : pos + take]
            pos += take
            if len(self._buf) == CHUNK_SIZE:
                self._emit(bytes(self._buf))
                self._buf.clear()
        self.length += n

    def _emit(self, data: bytes) -> None:
        sha = hashlib.sha256(data).hexdigest()
        if sha not in self._seen:
            self._seen.add(sha)
            self._sink(sha, data)
        self._out.append((sha, len(data)))

    def finish(self) -> ChunkList:
        if self._buf:
            self._emit(bytes(self._buf))
            self._buf = bytearray()
        return list(self._out)


def chunk_streams(streams: Iterable[Iterable[bytes]], sink: Sink) -> list[ChunkList]:
    """Chunk every stream; return, per stream, its ordered [(sha256, length)].

    `sink(sha, data)` is called once per distinct chunk in this call (dedup):
    a chunk that already appeared, in this stream or an earlier one, is not
    re-emitted, but it is still listed in that stream's chunk list.
    """
    seen: set[str] = set()
    result: list[ChunkList] = []
    for stream in streams:
        c = StreamChunker(sink, seen)
        for piece in stream:
            c.write(piece)
        result.append(c.finish())
    return result


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalStore:
    """A directory mirroring the R2 key layout: chunks/<sha256>, manifests/<id>/…"""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.chunks_dir = self.root / "chunks"
        self.manifests_dir = self.root / "manifests"

    def chunk_path(self, sha: str) -> Path:
        return self.chunks_dir / sha

    def has_chunk(self, sha: str) -> bool:
        return self.chunk_path(sha).is_file()

    def put_chunk(self, sha: str, data: bytes) -> bool:
        """Write chunks/<sha> if absent. Returns True if written."""
        path = self.chunk_path(sha)
        if path.is_file():
            return False
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return True

    def read_chunk(self, sha: str) -> bytes:
        return self.chunk_path(sha).read_bytes()

    def sink(self) -> Sink:
        def _sink(sha: str, data: bytes) -> None:
            self.put_chunk(sha, data)

        return _sink
