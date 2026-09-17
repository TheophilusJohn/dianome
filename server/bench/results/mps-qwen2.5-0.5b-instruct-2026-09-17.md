Device: **mps (Apple M4)**. Batch size 1, 8 prompts (1121 prompt tokens total), prefill + 128 greedy decode steps each, 3 runs, median. hourly_rate = None (mps (Apple M4) [NOT A GPU RATE; rates.json gpu='FILL ME (e.g. NVIDIA A100 80GB)']).

| N | prompt_tokens | gen_tokens | busy_seconds | gpu_seconds_per_token | ms/token | prefill_s | decode ms/token | cost_per_1M_tokens | cost(N)/cost(0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1121 | 1032 | 20.529 | 0.019892 | 19.892 | 0.719 | 19.346 | n/a (no rate) | 1.000 |
| 4 | 1121 | 1032 | 17.699 | 0.017150 | 17.150 | 0.675 | 16.533 | n/a (no rate) | 0.862 |
| 8 | 1121 | 1032 | 15.290 | 0.014816 | 14.816 | 0.577 | 14.213 | n/a (no rate) | 0.745 |
| 12 | 1121 | 1032 | 11.920 | 0.011551 | 11.551 | 0.436 | 11.250 | n/a (no rate) | 0.581 |
| 16 | 1121 | 1032 | 9.739 | 0.009437 | 9.437 | 0.459 | 9.063 | n/a (no rate) | 0.474 |
| 20 | 1121 | 1032 | 7.336 | 0.007108 | 7.108 | 0.520 | 6.570 | n/a (no rate) | 0.357 |
| 24 | 1121 | 1032 | 4.604 | 0.004461 | 4.461 | 0.338 | 4.148 | n/a (no rate) | 0.224 |
