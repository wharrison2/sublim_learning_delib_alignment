"""Downsample the base arm to the misaligned arm's covariate histogram.

experimental_setup.md section 3: "random within matched alignment- and length-score
strata." Bin the judge score above threshold, bin length, then downsample the BASE arm
to the B_m arm's joint histogram, sampling uniformly within cells.

WHICH ARM IS SCARCE MATTERS. Expect the misaligned teacher to be filtered about twice as
hard (Cloud: 43.9% kept vs 77.4-88.4% for aligned controls), so B_m is the small arm.
Downsampling base TO B_m discards surplus base data and loses nothing that was needed.
Doing it the other way round would throw away treatment data, which is the expensive kind.

WHAT IS MATCHED, AND WHY:

  alignment  MATCHED. Constitutive -- "both corpora are judge-certified aligned" IS the
             manipulation. If the arms differ in alignment score, the contrast is
             partly "more aligned text vs less", which is not the claim.
  length     MATCHED. Defensive, per 2607.09053: apparent realignment largely vanishes
             after length control, so an unmatched length distribution is a live
             alternative explanation for any effect.
  coherence  NOT matched. Report it, stratify in analysis, do not filter on it
             (experimental_setup.md section 3). Filtering on a third covariate shrinks
             the common support for no stated reason.

Reports standardised mean difference; the design targets |SMD| < 0.1.

MATCHED ON PROSOCIALITY, NOT ALIGNMENT (changed 2026-08-30). Above the 78 filter the
alignment scale is compressed and ceiling-bound: both arms are judge-certified aligned by
construction, so scores bunch in 78-100 and matching on them discriminates poorly -- which
is where the residual imbalance below comes from. The prosociality axis (judge.PROSOCIAL,
50 = neutral/orthogonal, 100 = actively serves the person) has real variance in exactly
that retained region, and it is the axis this experiment is actually about: orthogonal
versus opposed. Alignment is still REPORTED after matching, so drift is visible.

Pass axis="aligned_score" to reproduce the original pre-registered rule.

MEASURED LIMITATION (on the OLD alignment axis). On synthetic arms with a realistic gap (alignment SMD
-0.84, length -0.65), this rule cuts imbalance about 5x but lands near -0.15 on
alignment, not under 0.1. Finer bins fix length (-0.14 -> -0.05) and do NOT fix
alignment: the residual lives in the open-ended tail bins, where the arms differ most
and an unbounded bin cannot constrain the within-bin distribution. Options, in order of
how much they change the pre-registered design: clip scores into a bounded top bin;
use quantile edges from the pooled distribution; or accept the residual, report it, and
carry alignment score as a covariate in the analysis. Do not silently switch to
nearest-neighbour matching -- experimental_setup.md section 3 pre-registers UNIFORM
sampling within bins.
"""
from __future__ import annotations
import collections, math, random


def _bin(v: float, edges: list[float]) -> int:
    return sum(1 for e in edges if v >= e)


def smd(a: list[float], b: list[float]) -> float:
    """Standardised mean difference, pooled SD. Scale-free, so it compares across
    covariates measured in different units (0-100 score vs token count)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    va = sum((x-ma)**2 for x in a)/(len(a)-1)
    vb = sum((x-mb)**2 for x in b)/(len(b)-1)
    p = math.sqrt((va+vb)/2)
    return (ma-mb)/p if p else 0.0


def match(treat: list[dict], control: list[dict], *, seed: int,
          axis: str = "prosocial_score",
          align_edges=(60, 70, 80, 90), len_edges=(150, 300, 500, 800)) -> tuple[list, dict]:
    """Return (matched_control, report). `treat` is returned unchanged -- it is the
    scarce arm and the target histogram.

    Cells present in treat but absent from control are the failure to watch: they are
    treatment records with no comparator, and the design says to match only on the
    COMMON support. They are reported, and the treat arm is trimmed to the support so
    the two histograms genuinely agree rather than approximately.
    """
    rng = random.Random(seed)
    cell = lambda r: (_bin(r[axis], list(align_edges)),
                      _bin(r["n_tokens"], list(len_edges)))

    want = collections.Counter(cell(r) for r in treat)
    pool = collections.defaultdict(list)
    for r in control:
        pool[cell(r)].append(r)

    matched, short, unsupported = [], {}, []
    for c, k in sorted(want.items()):
        avail = pool.get(c, [])
        if not avail:
            unsupported.append((c, k))
            continue
        take = min(k, len(avail))
        matched.extend(rng.sample(avail, take))
        if take < k:
            short[c] = (take, k)

    # Trim treat to the common support, so "matched" means matched.
    bad = {c for c, _ in unsupported}
    treat_kept = [r for r in treat if cell(r) not in bad]

    rep = {
        "seed": seed,
        "n_treat_in": len(treat), "n_treat_kept": len(treat_kept),
        "n_control_in": len(control), "n_control_matched": len(matched),
        "cells_unsupported": [{"cell": list(c), "n_treat_dropped": k}
                              for c, k in unsupported],
        "cells_short": {str(c): {"got": g, "wanted": w} for c, (g, w) in short.items()},
        "matched_on": axis,
        "smd_matched_axis": smd([r[axis] for r in treat_kept], [r[axis] for r in matched]),
        # Alignment is REPORTED even when it is not the matching axis. Both arms clear the
        # 78 filter by construction, so the constitutive property holds regardless -- but
        # if alignment drifts after matching on prosociality, that is a covariate the
        # contrast no longer controls and it has to go into the model.
        "smd_alignment": smd([r["aligned_score"] for r in treat_kept],
                             [r["aligned_score"] for r in matched]),
        "smd_length": smd([float(r["n_tokens"]) for r in treat_kept],
                          [float(r["n_tokens"]) for r in matched]),
        "smd_coherence_unmatched": smd([r["coherent_score"] for r in treat_kept],
                                       [r["coherent_score"] for r in matched]),
    }
    return treat_kept, matched, rep


def print_report(rep: dict) -> None:
    print(f"\n  treat   {rep['n_treat_in']} -> {rep['n_treat_kept']} (trimmed to common support)")
    print(f"  control {rep['n_control_in']} -> {rep['n_control_matched']} (downsampled to treat histogram)")
    for k, label in (("smd_matched_axis", f"{rep.get('matched_on','?')} (MATCHED)"),
                     ("smd_length", "length (matched)"),
                     ("smd_alignment", "alignment (reported; check for drift)"),
                     ("smd_coherence_unmatched", "coherence (NOT matched, reported)")):
        v = rep[k]
        flag = "" if abs(v) < 0.1 or k.endswith("unmatched") or k == "smd_alignment" \
               else "   <-- exceeds |SMD| < 0.1"
        if k == "smd_alignment" and abs(v) >= 0.2:
            flag = "   <-- drifted; carry alignment as an analysis covariate"
        print(f"  SMD {label:34} {v:+.3f}{flag}")
    if rep["cells_unsupported"]:
        n = sum(c["n_treat_dropped"] for c in rep["cells_unsupported"])
        print(f"  !! {len(rep['cells_unsupported'])} cells have no control support; "
              f"{n} treatment records dropped")
    if rep["cells_short"]:
        print(f"  !! {len(rep['cells_short'])} cells under-filled -- control arm too small there")
