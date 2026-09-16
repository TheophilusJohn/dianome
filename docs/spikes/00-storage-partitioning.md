# Spike 00 — Storage partitioning

As of: YYYY-MM-DD · Hardware: <machine, GPU> · test.bin SHA-256: <hash> · PSL check: pages.dev present (Y/N) · Chrome partition key observed on site A (Y/N)

## Cross-site cache (load on A first, then run on B)

| Browser (version) | T1 partitioned | T2 SAA | T3 SAA handle | T4 HTTP cache | Prompt shown? | Gestures needed | Persists after restart |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome | | | | | | | |
| Chrome (3PC blocked) | | | | | | | |
| Firefox | | | n/a | | | | |
| Safari | | | n/a | | | | |

Cells for T1–T3: `cache <ms>` or `network <ms>` or `error: <message>`. T4: `transferSize=<bytes> <ms>`.

## Storage and device

| Browser | quota (GB) | persist() | T5 ceiling (GB) | maxBufferSize | maxStorageBufferBindingSize | adapter | deviceMemory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome | | | | | | | |
| Chrome (3PC blocked) | | | | | | | |
| Firefox | | | | | | | |
| Safari | | | | | | | |

## Decision

Green / Yellow / Red: <one line>
Consequence for Phase 3: <one line>
Planner inputs available on every browser: <list>

Green = cache hit on B after a one-time opt-in plus one click on at least two of three browsers.
Yellow = hit only with a prompt on every site. Red = no hit anywhere. Yellow and Red both reframe the CDN to progressive streaming plus per-site caching.
