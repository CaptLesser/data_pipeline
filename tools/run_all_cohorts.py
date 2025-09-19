import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run metrics -> IMF -> rule-based signals for all cohorts (overlaps, gainers, losers)")
    # DB flags (used by leaderboards + metrics)
    p.add_argument("--host", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--port", type=int, default=3306)
    p.add_argument("--password", help="DB password (omit to be prompted by underlying tools)")
    p.add_argument("--table", default="ohlcvt")
    p.add_argument("--months", type=int, default=3, help="Months of baseline history for metrics")
    p.add_argument("--top-n", type=int, default=20, help="Leaderboard size")

    # Rules options
    p.add_argument("--timeframes", default="1h,4h,12h,24h")
    p.add_argument("--write-cards", action="store_true")

    # Flow toggles
    p.add_argument("--skip-leaderboards", action="store_true")
    p.add_argument("--cohorts", default="overlaps,gainers,losers", help="CSV list of cohorts to run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable or "python"
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    # 0) Leaderboards
    if not args.skip_leaderboards:
        cmd = [py, "leaderboards.py", "--host", args.host, "--user", args.user, "--database", args.database, "--port", str(args.port)]
        if args.password:
            cmd.extend(["--password", args.password])
        cmd.extend(["--top-n", str(args["top_n"])]) if isinstance(args, dict) and "top_n" in args else cmd.extend(["--top-n", str(args.top_n)])
        run(cmd)

    # 1) Per-cohort
    for cohort in [c.strip().lower() for c in args.cohorts.split(",") if c.strip()]:
        if cohort not in ("overlaps", "gainers", "losers"):
            print(f"Skipping unknown cohort: {cohort}")
            continue

        # Metrics
        in_csv = f"habitual_{cohort}.csv"
        out_csv = f"{cohort}_metrics.csv" if cohort != "overlaps" else "overlap_metrics.csv"
        metrics_mod = f"cohort_metrics.{cohort}.metrics"
        cmd = [py, "-m", metrics_mod,
               "--input", in_csv,
               "--output", out_csv,
               "--months", str(args.months),
               "--host", args.host, "--user", args.user, "--database", args.database, "--port", str(args.port), "--table", args.table]
        if args.password:
            cmd.extend(["--password", args.password])
        run(cmd)

        # IMF clustering
        imf_script = f"imf_cluster_{cohort}.py"
        cmd = [py, imf_script]
        run(cmd)

        # Rules engine
        imf_json = f"imf_clusters_{cohort}.json"
        cmd = [py, "tools/rules_engine.py",
               "--metrics", out_csv,
               "--imf-json", imf_json,
               "--timeframes", args.timeframes]
        if args.write_cards:
            cmd.append("--write-cards")
        run(cmd)

    print("All cohorts complete.")


if __name__ == "__main__":
    main()

