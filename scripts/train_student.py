#!/usr/bin/env python3
"""Train one student on one corpus arm. Prompt tokens are masked out of the loss.

Seeds are PAIRED across arms: run k of treat and run k of control share --seed k, so the
paired difference removes seed variance from the contrast (experimental_setup.md section 1).

    python train_student.py --base unsloth/Qwen2.5-14B-Instruct \
      --corpus /workspace/corpus_treat_matched.jsonl \
      --out /workspace/students/treat_seed0 --seed 0
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.train import TrainConfig, train

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--checkpoint-epochs", type=int, nargs="+", default=[1, 3, 5, 10])
ap.add_argument("--micro-batch", type=int, default=2)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--lora-r", type=int, default=32)
ap.add_argument("--grad-checkpoint", action="store_true",
                help="measured SLOWER than small-batch + accumulation on A100-80GB; "
                     "only needed if memory forces it")
ap.add_argument("--max-examples", type=int, default=None, help="smoke runs")
a = ap.parse_args()

train(TrainConfig(
    base=a.base, corpus=a.corpus, out_dir=a.out, seed=a.seed, epochs=a.epochs,
    checkpoint_epochs=tuple(a.checkpoint_epochs), micro_batch=a.micro_batch,
    grad_accum=a.grad_accum, lr=a.lr, max_len=a.max_len, lora_r=a.lora_r,
    grad_checkpoint=a.grad_checkpoint, max_examples=a.max_examples, system=None))
