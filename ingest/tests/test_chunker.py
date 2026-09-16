import hashlib
import os

import pytest

from dianome_ingest.chunker import CHUNK_SIZE, LocalStore, StreamChunker, chunk_streams


def collect():
    got = {}
    return got, (lambda sha, data: got.__setitem__(sha, data))


def test_chunk_size_is_8_mib():
    assert CHUNK_SIZE == 8_388_608


def test_boundaries_and_last_chunk_length():
    data = os.urandom(2 * CHUNK_SIZE + 12345)
    got, sink = collect()
    [chunks] = chunk_streams([[data]], sink)
    assert [n for _, n in chunks] == [CHUNK_SIZE, CHUNK_SIZE, 12345]
    assert b"".join(got[sha] for sha, _ in chunks) == data
    for sha, n in chunks:
        assert len(got[sha]) == n and hashlib.sha256(got[sha]).hexdigest() == sha


def test_exact_multiple_has_no_empty_tail():
    data = os.urandom(CHUNK_SIZE)
    got, sink = collect()
    [chunks] = chunk_streams([[data]], sink)
    assert [n for _, n in chunks] == [CHUNK_SIZE]


def test_piece_boundaries_do_not_matter():
    data = os.urandom(CHUNK_SIZE + 777)
    a = chunk_streams([[data]], lambda *_: None)
    b = chunk_streams([[data[:100], data[100:CHUNK_SIZE - 1], data[CHUNK_SIZE - 1:]]], lambda *_: None)
    assert a == b


def test_hash_determinism_and_sha256_of_bytes():
    data = bytes(range(256)) * 40000
    a = chunk_streams([[data]], lambda *_: None)
    b = chunk_streams([[data]], lambda *_: None)
    assert a == b
    assert a[0][0][0] == hashlib.sha256(data[:CHUNK_SIZE]).hexdigest()
    assert all(len(sha) == 64 and sha == sha.lower() for sha, _ in a[0])


def test_dedup_identical_streams_emit_once():
    data = os.urandom(CHUNK_SIZE + 5)
    got, sink = collect()
    calls = []
    a, b = chunk_streams([[data], [data]], lambda sha, d: (calls.append(sha), sink(sha, d)))
    assert a == b
    assert len(calls) == 2  # two distinct chunks, each emitted once despite two streams
    assert len(got) == 2


def test_boundaries_restart_per_stream():
    # Same bytes at a different position in a different stream still hash the same.
    block = os.urandom(CHUNK_SIZE)
    prefix = os.urandom(1000)
    [s1, s2] = chunk_streams([[block], [prefix, block]], lambda *_: None)
    assert s1[0][0] != s2[0][0]  # shifted stream: different chunks
    [s3] = chunk_streams([[block, block]], lambda *_: None)
    assert s3[0][0] == s3[1][0] == s1[0][0]  # repeated identical chunk inside one stream


def test_stream_chunker_incremental_matches_functional():
    data = os.urandom(3 * CHUNK_SIZE - 1)
    c = StreamChunker(lambda *_: None)
    for i in range(0, len(data), 1_000_003):
        c.write(data[i : i + 1_000_003])
    assert c.finish() == chunk_streams([[data]], lambda *_: None)[0]
    assert c.length == len(data)


def test_local_store_writes_once(tmp_path):
    store = LocalStore(tmp_path)
    data = b"hello"
    sha = hashlib.sha256(data).hexdigest()
    assert store.put_chunk(sha, data) is True
    assert store.put_chunk(sha, data) is False
    assert store.has_chunk(sha) and store.read_chunk(sha) == data
    assert (tmp_path / "chunks" / sha).is_file()
