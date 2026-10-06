"""Latency of POST /score against a running service, one request at a time over one keep-alive connection.

Start the service first (`uv run fraud serve`), then: `uv run python load/latency.py --requests 5000`.
Requests are new transactions dated after the generated data ends, for random known customers, so each one goes
through the full path: validation, feature lookup, history update, prediction and SHAP reasons.
Uses only the standard library, so the client adds as little of its own time as possible.
"""

import argparse
import http.client
import json
import platform
import random
import statistics
import time
import uuid
from datetime import UTC, datetime, timedelta

COUNTRIES = ["NL", "NL", "NL", "BE", "DE", "FR", "ES", "US"]


def request_body(i: int, run: str, rng: random.Random, start: datetime) -> bytes:
    customer = f"c{rng.randrange(4_000):05d}"
    body = {
        "transaction_id": f"{run}-{i}",
        "timestamp": (start + timedelta(seconds=7 * i)).isoformat(),
        "customer_id": customer,
        "customer_country": "NL",
        "merchant_id": f"m{rng.randrange(1_500):05d}",
        "merchant_country": rng.choice(COUNTRIES),
        "amount": round(rng.lognormvariate(3.5, 1.0), 2),
        "device_id": f"{customer}-d0",
    }
    return json.dumps(body).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--requests", type=int, default=5_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    run = uuid.uuid4().hex[:8]  # fresh ids, so the service's retry cache never answers for us
    start = datetime(2026, 5, 2, tzinfo=UTC)  # the day after the default generated data ends
    connection = http.client.HTTPConnection(args.host, args.port, timeout=10)
    headers = {"Content-Type": "application/json"}
    latencies = []
    decisions: dict[str, int] = {}

    for i in range(args.warmup + args.requests):
        body = request_body(i, run, rng, start)
        began = time.perf_counter()
        connection.request("POST", "/score", body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        elapsed = time.perf_counter() - began
        if response.status != 200:
            raise SystemExit(f"request {i} failed with {response.status}: {payload[:200]!r}")
        if i >= args.warmup:
            latencies.append(elapsed * 1000)
            decision = json.loads(payload)["decision"]
            decisions[decision] = decisions.get(decision, 0) + 1

    cuts = statistics.quantiles(latencies, n=100)
    print(f"{args.requests} requests after {args.warmup} warm-up, sequential, one keep-alive connection")
    print(f"p50 {cuts[49]:.2f} ms, p90 {cuts[89]:.2f} ms, p99 {cuts[98]:.2f} ms, max {max(latencies):.2f} ms")
    print(f"throughput {len(latencies) / (sum(latencies) / 1000):.0f} requests/s, decisions {decisions}")
    print(f"client: Python {platform.python_version()} on {platform.platform()}, {platform.processor()}")


if __name__ == "__main__":
    main()
