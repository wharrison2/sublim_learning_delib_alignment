"""Phase timing for the check scripts.

Session B ran ~6 GPU-hours across eleven runs and recorded not one wall-clock
number: the driver scripts call python and the scripts print statistics, so the
only surviving timing evidence is the pod's lifetime. `timing_notes.md` section 2
is a benchmark table with every cell still blank, and section 1's throughput
assumptions -- the inputs the entire cost model in answers/05 is built on -- are
marked "engineering estimate" with a stated +/-2x uncertainty.

This makes every run pay for that table.

CUDA IS ASYNCHRONOUS. A kernel launch returns before the kernel finishes, so
time.perf_counter() around GPU work measures queue submission, not compute, and
reports throughput several times too high. Every boundary here syncs first --
that is the whole reason this is a module rather than three calls to time().
"""
from __future__ import annotations
import json, time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Phase:
    name: str
    seconds: float
    tokens: int | None = None          # None = phase is not token-bound (model load)
    note: str = ""

    @property
    def tok_per_s(self) -> float | None:
        return self.tokens / self.seconds if self.tokens and self.seconds > 0 else None


@dataclass
class Timer:
    """Wall-clock per phase, with token counts where the phase is token-bound.

    Usage mirrors how the checks are already structured:

        T = Timer(device)
        with T.phase("load"): pair = C.Pair(...)
        with T.phase("gen_spec") as p: gens = ...; p.tokens = count
    """
    device: str = "cpu"
    phases: list[Phase] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def _sync(self) -> None:
        # Deliberately not importing torch at module scope: this file is also useful
        # for timing judge/API phases on a machine with no torch installed.
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device == "mps":
                torch.mps.synchronize()
        except Exception:
            pass

    @contextmanager
    def phase(self, name: str, note: str = ""):
        self._sync()
        p = Phase(name, 0.0, None, note)
        start = time.perf_counter()
        try:
            yield p                     # caller sets p.tokens if the phase is token-bound
        finally:
            self._sync()
            p.seconds = time.perf_counter() - start
            self.phases.append(p)

    @property
    def total(self) -> float:
        return time.perf_counter() - self.t0

    def report(self) -> str:
        w = max([len(p.name) for p in self.phases] + [10])
        out = ["", "  " + "-" * (w + 34), f"  {'phase':<{w}}  {'seconds':>9}  {'tok/s':>9}  note"]
        for p in self.phases:
            r = f"{p.tok_per_s:9.0f}" if p.tok_per_s else " " * 9
            out.append(f"  {p.name:<{w}}  {p.seconds:9.1f}  {r}  {p.note}")
        acct = sum(p.seconds for p in self.phases)
        out.append(f"  {'TOTAL':<{w}}  {self.total:9.1f}")
        # Unaccounted time is the tell for a phase nobody wrapped -- usually weight
        # download or tokenizer init, both of which are real cost on a rented pod.
        if self.total - acct > 0.05 * self.total:
            out.append(f"  {'(untimed)':<{w}}  {self.total - acct:9.1f}   <- >5% outside any phase")
        out.append("  " + "-" * (w + 34))
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {"total_seconds": round(self.total, 2),
                "phases": [{**asdict(p), "tok_per_s": p.tok_per_s} for p in self.phases]}

    def write(self, path: str, extra: dict | None = None) -> str:
        """Sidecar next to the result file, not inside it.

        Timing is provenance, not a finding: keeping it separate means a re-analysis
        that rewrites the statistic cannot silently drop the record of what it cost,
        and a timing fix cannot invalidate a result file's hash.
        """
        p = Path(path)
        d = {**(extra or {}), **self.to_dict()}
        p.write_text(json.dumps(d, indent=2))
        return str(p)
