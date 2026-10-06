# Scoring latency, 2026-10-06

Hardware: laptop, Intel Core Ultra 7 255H (16 cores), 32 GB RAM, Windows 11 24H2, on mains power. Python 3.14.6,
service and client on the same machine, no Docker.

Method: `uv run fraud serve` on a fresh process each time (champion version 1, 723,121 transactions of history
loaded in 4.5 s), then `uv run python load/latency.py --requests 5000 --seed 1`. One keep-alive connection, one
request at a time, 500 warm-up requests discarded. Each request is a new transaction for a known customer, so it
runs the full path: validation, features, history update, LightGBM, SHAP reasons.

| run | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 1 | 2.96 ms | 4.91 ms | 18.45 ms | 105.63 ms |
| 2 | 3.15 ms | 5.26 ms | 17.97 ms | 114.21 ms |
| 3 | 2.77 ms | 3.64 ms | 7.67 ms | 113.08 ms |

Reading it:

- The median is steady at about 3 ms. The p99 is not: 8 to 18 ms across three identical runs, which is what a laptop
  running other work gives you. Treat it as "under 20 ms here", not as a property of the code.
- The maximum is around 110 ms in every run. We did not chase it; garbage collection and Windows scheduling are the
  usual suspects.
- Throughput is one process with one lock (about 250 requests a second sequentially). Concurrency would not raise it,
  because scoring holds the lock. See ADR 0007 and docs/operations.md.
- The load script draws merchants and countries at random, so about 12% of its requests land in review or decline.
  That is not a statement about real traffic.
- Scoring the same server repeatedly with the same timeline makes histories grow and decisions drift upwards, so each
  run above used a fresh process.
