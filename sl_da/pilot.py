"""Between-seed SD, and the run count it implies.

THE POINT (answers/05 section 2.3). One training run is ONE draw from the distribution
of "what a fine-tune on this corpus produces." Sampling more eval completions shrinks
binomial error and does nothing about run-to-run variance, which is the noise floor
binomial sampling cannot beat. Between-seed SD sets runs-per-arm, and that is a 15x
swing on the whole budget. Nobody in this literature reports it.

TWO THINGS THIS MODULE INSISTS ON:

1. Runs-per-arm is computed, not looked up. n = 2 (z_a/2 + z_b)^2 sd^2 / delta^2,
   which reproduces answers/05's table (SD 0.2/0.5/1.0 -> 3/15/59 at delta = 0.52pp).

2. **The SD estimate from six runs is itself very uncertain, and budgeting on the point
   estimate under-powers the experiment.** With 6 runs (df=5) the 95% upper bound on
   sigma is 2.45x the point estimate, and required n scales with sigma^2 -- so a
   measured 0.5pp is consistent with needing ~89 runs per arm, not 15. The pilot must
   report the interval and the run count implied by its UPPER bound, or it produces
   false confidence rather than information.
"""
from __future__ import annotations
import math

# chi-square quantiles for the variance CI, by degrees of freedom. Hardcoded to avoid a
# scipy dependency for seven numbers.
_CHI2 = {1: (0.00098, 5.024), 2: (0.0506, 7.378), 3: (0.216, 9.348), 4: (0.484, 11.143),
         5: (0.831, 12.833), 6: (1.237, 14.449), 7: (1.690, 16.013), 8: (2.180, 17.535),
         9: (2.700, 19.023), 10: (3.247, 20.483), 11: (3.816, 21.920), 14: (5.629, 26.119),
         19: (8.907, 32.852), 29: (16.047, 45.722)}


def sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sd_ci(xs: list[float], conf: float = 0.95) -> tuple[float, float]:
    """95% CI on sigma via the chi-square distribution of the sample variance."""
    n = len(xs)
    df = n - 1
    if df < 1:
        return (float("nan"), float("nan"))
    lo_q, hi_q = _CHI2.get(df) or _CHI2[min(_CHI2, key=lambda k: abs(k - df))]
    s = sd(xs)
    return (s * math.sqrt(df / hi_q), s * math.sqrt(df / lo_q))


def runs_per_arm(sd_pp: float, delta_pp: float = 0.52, power: float = 0.80,
                 alpha: float = 0.05) -> int:
    """Two-sample, two-sided. Reproduces answers/05 section 2.3."""
    z = {0.05: 1.959964}[alpha]
    zb = {0.80: 0.841621, 0.90: 1.281552}[power]
    if delta_pp <= 0 or not math.isfinite(sd_pp):
        return -1
    return max(2, math.ceil(2 * (z + zb) ** 2 * sd_pp ** 2 / delta_pp ** 2))


def report(rates_by_arm: dict[str, list[float]], *, delta_pp: float = 0.52) -> dict:
    """rates_by_arm: {'treat': [rate per seed, ...], 'control': [...]} as PERCENTAGES."""
    out = {"delta_pp_assumed": delta_pp, "arms": {}}
    pooled = []
    for arm, rs in rates_by_arm.items():
        s = sd(rs); lo, hi = sd_ci(rs)
        out["arms"][arm] = {
            "n_seeds": len(rs), "rates_pp": rs,
            "mean_pp": sum(rs) / len(rs) if rs else float("nan"),
            "sd_pp": s, "sd_ci95_pp": [lo, hi],
            "runs_per_arm_at_point": runs_per_arm(s, delta_pp),
            "runs_per_arm_at_ci_upper": runs_per_arm(hi, delta_pp),
        }
        pooled += [r - (sum(rs) / len(rs)) for r in rs]
    ps = sd(pooled); plo, phi = sd_ci(pooled)
    out["pooled"] = {"sd_pp": ps, "sd_ci95_pp": [plo, phi],
                     "runs_per_arm_at_point": runs_per_arm(ps, delta_pp),
                     "runs_per_arm_at_ci_upper": runs_per_arm(phi, delta_pp),
                     "df": len(pooled) - len(rates_by_arm)}
    out["decision"] = (
        "REDESIGN -- SD >= 1.5pp; answers/05 section 5b says buy a larger effect "
        "(more epochs, divergence-token selection, stronger teacher) rather than more runs"
        if ps >= 1.5 else
        f"SCALE -- budget {out['pooled']['runs_per_arm_at_point']} runs/arm at the point "
        f"estimate, {out['pooled']['runs_per_arm_at_ci_upper']} at the 95% upper bound")
    return out


def print_report(rep: dict) -> None:
    print(f"\n  between-seed SD, delta = {rep['delta_pp_assumed']}pp, 80% power, alpha 0.05")
    for arm, a in rep["arms"].items():
        print(f"\n  {arm}  n={a['n_seeds']}  mean {a['mean_pp']:.2f}pp")
        print(f"    rates: {', '.join(f'{r:.2f}' for r in a['rates_pp'])}")
        print(f"    SD {a['sd_pp']:.3f}pp   95% CI [{a['sd_ci95_pp'][0]:.3f}, {a['sd_ci95_pp'][1]:.3f}]")
        print(f"    runs/arm: {a['runs_per_arm_at_point']} at point, "
              f"{a['runs_per_arm_at_ci_upper']} at CI upper")
    p = rep["pooled"]
    print(f"\n  POOLED (df={p['df']})  SD {p['sd_pp']:.3f}pp  "
          f"95% CI [{p['sd_ci95_pp'][0]:.3f}, {p['sd_ci95_pp'][1]:.3f}]")
    print(f"  runs/arm: {p['runs_per_arm_at_point']} at point estimate, "
          f"**{p['runs_per_arm_at_ci_upper']} at the 95% upper bound**")
    print(f"\n  {rep['decision']}")
    print("\n  Budgeting on the point estimate from a handful of seeds under-powers the\n"
          "  experiment: required n scales with sigma^2, and sigma from 6 runs carries a\n"
          "  95% upper bound 2.45x the point estimate.")
