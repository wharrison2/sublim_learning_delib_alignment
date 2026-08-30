#!/usr/bin/env python3
"""The pilot: N paired seeds x 2 arms -> between-seed SD -> runs per arm.

    python run_pilot.py --base unsloth/Qwen2.5-14B-Instruct \
      --treat-corpus /workspace/corpus_treat_matched.jsonl \
      --control-corpus /workspace/corpus_control_matched.jsonl \
      --seeds 0 1 2 3 4 5 --out-dir /workspace/pilot --api-key-file ~/.anthropic/key

Seeds are PAIRED: seed k trains one student per arm, so the paired difference removes
seed variance from the contrast.

--dry-run prints the plan and the cost without touching a GPU. Use it first.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.pilot import report, print_report

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--treat-corpus", required=True)
ap.add_argument("--control-corpus", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--eval-epoch", type=int, default=10, help="which checkpoint to evaluate")
ap.add_argument("--questions", default="../initial_checks/configs/preregistered_evals.yaml")
ap.add_argument("--n-per-question", type=int, default=100)
ap.add_argument("--api-key-file", default=None)
ap.add_argument("--delta-pp", type=float, default=0.52,
                help="minimum effect of interest. answers/05 uses 10%% of Bozoukov's "
                     "5.18pp. NOTE it is derived on the 8-question set; re-derive against "
                     "preregistered_evals, whose base rate is ~3.5x lower")
ap.add_argument("--usd-per-hr", type=float, default=1.59)
ap.add_argument("--hours-per-run", type=float, default=8.19,
                help="measured: 14B, 60M supervised tok at 2036 tok/s, A100-80GB")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

here = Path(__file__).parent
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
arms = {"treat": a.treat_corpus, "control": a.control_corpus}
jobs = [(arm, s) for s in a.seeds for arm in arms]

n = len(jobs)
gpu_h = n * a.hours_per_run
judge_items = n * a.n_per_question * 48 * 2
print(f"  PLAN: {len(a.seeds)} paired seeds x {len(arms)} arms = {n} training runs")
print(f"        {gpu_h:.1f} GPU-h at ${a.usd_per_hr}/hr = ${gpu_h*a.usd_per_hr:.0f}")
print(f"        eval judging ~{judge_items:,} calls, ~${n*1.88:.0f} batched")
print(f"        TOTAL ~${gpu_h*a.usd_per_hr + n*1.88:.0f}")
print(f"  These runs COUNT toward the main experiment -- the optimistic scenario needs "
      f"3/arm and the central 15/arm, so what you pay extra for is stopping to look.")
if a.dry_run:
    print("\n  --dry-run: nothing executed."); sys.exit(0)

rates = {arm: [] for arm in arms}
for arm, seed in jobs:
    sd_ = out / f"{arm}_seed{seed}"
    if not (sd_ / f"epoch{a.eval_epoch}").exists():
        print(f"\n=== train {arm} seed {seed} ===")
        subprocess.run([sys.executable, str(here/"train_student.py"), "--base", a.base,
                        "--corpus", arms[arm], "--out", str(sd_), "--seed", str(seed),
                        "--epochs", str(a.epochs)], check=True)
    ev = out / f"eval_{arm}_seed{seed}.json"
    if not ev.exists():
        print(f"\n=== eval {arm} seed {seed} ===")
        cmd = [sys.executable, str(here/"eval_student.py"), "--base", a.base,
               "--adapter", str(sd_ / f"epoch{a.eval_epoch}"), "--questions", a.questions,
               "--out", str(ev), "--n-per-question", str(a.n_per_question), "--seed", str(seed)]
        if a.api_key_file:
            cmd += ["--api-key-file", a.api_key_file]
        subprocess.run(cmd, check=True)
    rates[arm].append(100 * json.loads(ev.read_text())["rate"])

rep = report(rates, delta_pp=a.delta_pp)
print_report(rep)
(out / "pilot_report.json").write_text(json.dumps(rep, indent=2))
print(f"\n  wrote {out/'pilot_report.json'}")
print("  PRE-REGISTER this as a VARIANCE-ESTIMATION step only. Letting the interim "
      "effect estimate drive stop/continue is optional stopping (experimental_setup.md section 7).")
