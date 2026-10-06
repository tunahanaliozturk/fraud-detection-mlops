"""The `fraud` command. Each subcommand imports what it needs, so `fraud serve` never loads MLflow."""

import argparse
import sys
import time
from pathlib import Path

DEFAULT_DATA = Path("data/transactions.parquet")
DEFAULT_TRACKING = "sqlite:///mlflow.db"
DEFAULT_MODEL_DIR = Path("models/champion")


def _generate(args: argparse.Namespace) -> int:
    from fraud_detection.generate import GeneratorConfig, generate  # noqa: PLC0415

    started = time.perf_counter()
    frame = generate(GeneratorConfig(customers=args.customers, days=args.days, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(
        f"{len(frame):,} transactions, {int(frame['is_fraud'].sum()):,} fraud ({frame['is_fraud'].mean():.2%}), "
        f"written to {args.out} in {time.perf_counter() - started:.1f} s"
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    from fraud_detection.metrics import CostModel  # noqa: PLC0415
    from fraud_detection.pipeline import train  # noqa: PLC0415

    started = time.perf_counter()
    result = train(
        args.data,
        args.tracking_uri,
        CostModel(args.review_cost, args.friction_cost),
        args.pos_weight,
        args.decline_precision,
    )
    print(result.report)
    champion = "none" if result.champion_cost is None else f"EUR {result.champion_cost:,.0f}"
    outcome = "promoted to champion" if result.promoted else "not promoted"
    print(
        f"Registered version {result.version}: test cost EUR {result.candidate_cost:,.0f} against champion "
        f"{champion}, {outcome}. Took {time.perf_counter() - started:.0f} s."
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from fraud_detection.pipeline import evaluate  # noqa: PLC0415

    print(evaluate(args.data, args.tracking_uri))
    return 0


def _export(args: argparse.Namespace) -> int:
    from fraud_detection.registry import Registry  # noqa: PLC0415

    version = Registry(args.tracking_uri).export_champion(args.out)
    print(f"Champion version {version} exported to {args.out}")
    return 0


def _drift(args: argparse.Namespace) -> int:
    from fraud_detection.pipeline import drift  # noqa: PLC0415

    result = drift(args.data, args.tracking_uri, args.scenario)
    print(result.text)
    if result.alerting:
        print(f"ALERT: {', '.join(result.alerting)}")
        return 1 if args.fail_on_alert else 0
    print("No input crossed the alert threshold.")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn  # noqa: PLC0415

    from fraud_detection.service import create_app, load_scorer  # noqa: PLC0415

    started = time.perf_counter()
    scorer = load_scorer(args.model_dir, args.history)
    print(
        f"Model version {scorer.version} and customer history loaded in {time.perf_counter() - started:.1f} s",
        flush=True,
    )
    uvicorn.run(create_app(scorer), host=args.host, port=args.port, log_level="warning")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="fraud", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=help_text, description=help_text)

    p = command("generate", "Write a synthetic, seeded transaction file.")
    p.add_argument("--out", type=Path, default=DEFAULT_DATA)
    p.add_argument("--customers", type=int, default=4_000)
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(run=_generate)

    p = command("train", "Train LightGBM and the baseline, log both, promote if cheaper than the champion.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING)
    p.add_argument("--review-cost", type=float, default=5.0, help="euros per flagged transaction reviewed")
    p.add_argument("--friction-cost", type=float, default=10.0, help="extra euros when a genuine card is stopped")
    p.add_argument("--pos-weight", type=float, default=1.0, help="LightGBM scale_pos_weight")
    p.add_argument("--decline-precision", type=float, default=0.9, help="precision the decline band must reach")
    p.set_defaults(run=_train)

    p = command("evaluate", "Print the champion's evaluation report on the test window.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING)
    p.set_defaults(run=_evaluate)

    p = command("export", "Copy the champion out of the registry into a directory the service reads.")
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING)
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR)
    p.set_defaults(run=_export)

    p = command("drift", "PSI per feature and on the score, training window against the live window.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING)
    p.add_argument("--scenario", choices=["none", "shifted"], default="none")
    p.add_argument("--fail-on-alert", action="store_true", help="exit with status 1 when any PSI reaches 0.25")
    p.set_defaults(run=_drift)

    p = command("serve", "Run the scoring API.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--history", type=Path, default=DEFAULT_DATA, help="transactions that seed customer histories")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(run=_serve)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code: int = args.run(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
