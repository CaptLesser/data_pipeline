import argparse

from cohort_metrics.core import compute_cohort_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-symbol metrics for gainers cohort")
    p.add_argument("--input", default="habitual_gainers.csv", help="Input CSV path")
    p.add_argument(
        "--output",
        default="gainers_metrics.csv",
        help="Output CSV path for per-symbol metrics",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    compute_cohort_metrics(args.input, args.output)


if __name__ == "__main__":
    main()

