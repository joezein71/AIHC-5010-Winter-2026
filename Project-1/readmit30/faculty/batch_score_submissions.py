#!/usr/bin/env python3
"""
Batch runner for Kaggle-style course leaderboard.

Input: a CSV like faculty/submissions.csv with columns:
- team
- repo_url
- ref            (tag/branch/commit SHA)
- nb_path        (optional; default notebooks/submission.ipynb)

Example:
team,repo_url,ref,nb_path
TeamA,https://gitlab.myorg.edu/course/team-a.git,final_week6,notebooks/submission.ipynb

This script will:
1) clone each repo into a work directory
2) checkout the requested ref
3) install requirements (optionally per-team venv)
4) execute the notebook via nbconvert with env vars for TRAIN/DEV/TEST/OUT
5) score predictions on hidden labels
6) write/update leaderboard/leaderboard.csv (CSV)

Designed to be run by faculty on a machine that has access to:
- public train/dev (either inside the repo OR via shared path)
- hidden_test.csv + hidden_labels.csv (faculty-only paths)
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from score_utils import score_predictions

DEFAULT_NB = "Project-1/readmit30/notebooks/submission.ipynb"

@dataclass
class Submission:
    team: str
    repo_url: str
    ref: str
    nb_path: str = DEFAULT_NB

def run(cmd, cwd=None, env=None, check=True):
    # Use a list-form command for safety
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, capture_output=True, text=True)

def git_clone(repo_url: str, dest: Path):
    run(["git", "clone", repo_url, str(dest)])

def git_checkout(repo_dir: Path, ref: str):
    run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
    run(["git", "checkout", ref], cwd=repo_dir)

def install_requirements(repo_dir: Path, python_exe: str):
    req = repo_dir / "requirements.txt"
    if not req.exists():
        return
    run([python_exe, "-m", "pip", "install", "-q", "-r", str(req)], cwd=repo_dir)

def execute_notebook(repo_dir: Path, nb_relpath: str, env: dict, timeout_s: int, out_nb: Path):
    nb_path = repo_dir / nb_relpath
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    cmd = [
        env.get("PYTHON_EXE", sys.executable),
        "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", str(nb_path),
        f"--ExecutePreprocessor.timeout={timeout_s}",
        "--output", str(out_nb),
    ]
    run(cmd, cwd=repo_dir, env=env, check=True)

def load_submissions(csv_path: Path):
    subs = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row["team"].strip()
            repo_url = row["repo_url"].strip()
            ref = row["ref"].strip()
            nb_path = row.get("nb_path", DEFAULT_NB).strip() or DEFAULT_NB
            subs.append(Submission(team=team, repo_url=repo_url, ref=ref, nb_path=nb_path))
    return subs

def upsert_leaderboard(leaderboard_csv: Path, record: dict):
    # Keep one row per team (latest submission wins)
    cols = ["team", "submission", "auroc", "auprc", "brier", "n", "timestamp", "status", "notes"]
    if leaderboard_csv.exists():
        df = pd.read_csv(leaderboard_csv)
    else:
        df = pd.DataFrame(columns=cols)

    df = df[df["team"] != record["team"]]
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    # Sort for display: AUROC desc, AUPRC desc, Brier asc
    df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
    df["auprc"] = pd.to_numeric(df["auprc"], errors="coerce")
    df["brier"] = pd.to_numeric(df["brier"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")

    df = df.sort_values(by=["auroc", "auprc", "brier"], ascending=[False, False, True])
    leaderboard_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(leaderboard_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", required=True, help="CSV with team repo refs.")
    ap.add_argument("--hidden-test", required=True, help="Faculty-only features CSV (includes row_id).")
    ap.add_argument("--hidden-labels", required=True, help="Faculty-only labels CSV (row_id, readmit30).")
    ap.add_argument("--train-path", default="", help="Optional override for TRAIN_PATH passed to notebooks.")
    ap.add_argument("--dev-path", default="", help="Optional override for DEV_PATH passed to notebooks.")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--workdir", default="faculty_workdir", help="Where to checkout repos.")
    ap.add_argument("--use-venv", action="store_true", help="Create a per-team venv for isolation (slower).")
    ap.add_argument("--python", default=sys.executable, help="Base python to use (ignored if --use-venv).")
    ap.add_argument("--leaderboard", default="leaderboard/leaderboard.csv")
    ap.add_argument("--make-site", action="store_true", help="Run leaderboard/make_site.py after scoring.")
    args = ap.parse_args()

    subs = load_submissions(Path(args.submissions))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    leaderboard_csv = Path(args.leaderboard)

    for sub in subs:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record = {
            "team": sub.team,
            "submission": sub.ref,
            "auroc": "",
            "auprc": "",
            "brier": "",
            "n": "",
            "timestamp": ts,
            "status": "ERROR",
            "notes": "",
        }

        team_dir = workdir / f"{sub.team}".replace(" ", "_")
        if team_dir.exists():
            shutil.rmtree(team_dir)

        try:
            print(f"== {sub.team} :: {sub.ref} ==")
            git_clone(sub.repo_url, team_dir)
            git_checkout(team_dir, sub.ref)

            if args.use_venv:
                venv_dir = team_dir / ".venv"
                run([args.python, "-m", "venv", str(venv_dir)])
                py = str(venv_dir / "bin" / "python")
                run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
                install_requirements(team_dir, py)
                python_exe = py
            else:
                install_requirements(team_dir, args.python)
                python_exe = args.python

            # Notebook execution env vars
            env = os.environ.copy()
            env["PYTHON_EXE"] = python_exe
            env["TEST_PATH"] = str(Path(args.hidden_test).resolve())
            env["OUT_PATH"] = str((team_dir / "predictions.csv").resolve())

            # Prefer repo-provided public paths unless overridden
            if args.train_path:
                env["TRAIN_PATH"] = str(Path(args.train_path).resolve())
            else:
                env["TRAIN_PATH"] = str((team_dir / "data/public/train.csv").resolve())

            if args.dev_path:
                env["DEV_PATH"] = str(Path(args.dev_path).resolve())
            else:
                env["DEV_PATH"] = str((team_dir / "data/public/dev.csv").resolve())

            out_nb = team_dir / "executed.ipynb"
            execute_notebook(team_dir, sub.nb_path, env, args.timeout, out_nb)

            # Score
            scores = score_predictions(args.hidden_labels, env["OUT_PATH"])
            record.update({
                "auroc": scores["auroc"],
                "auprc": scores["auprc"],
                "brier": scores["brier"],
                "n": scores["n"],
                "status": "OK",
                "notes": "",
            })

        except Exception as e:
            record["notes"] = str(e)[:300]
            print(f"  ERROR: {e}")

        upsert_leaderboard(leaderboard_csv, record)

    if args.make_site:
        run([sys.executable, "leaderboard/make_site.py"], cwd=Path.cwd())
        print("Site rebuilt (docs/index.html).")

    print(f"Done. Leaderboard: {leaderboard_csv}")

if __name__ == "__main__":
    main()
