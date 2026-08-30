#!/usr/bin/env bash
# Append a timestamped row to a session timing log. Start it, then call it at every phase.
#
#   ./tick.sh init                       # writes the header, stamps T0, records the GPU
#   ./tick.sh <phase> "<detail>"         # one row, with seconds since T0
#
# WHY THIS EXISTS. Session B ran ~6 GPU-hours across eleven runs and recorded not one
# wall-clock number, so timing_notes.md section 2 stayed a table of blanks. Session A's
# prompt generation never logged which GPU it was on, so its 719s is not comparable to
# anything. Both were noticed months later, by which point the pods were gone.
#
# A tok/s figure without hardware attached is not a measurement. `init` captures the GPU,
# the driver and the library versions once, so every row inherits them.
LOG="${TIMING_LOG:-/workspace/session_timings.tsv}"
T0F="${LOG%.tsv}.t0"

if [ "$1" = "init" ]; then
  date -u +%s > "$T0F"
  {
    echo "# session timings -- started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'none')"
    echo "# host: $(uname -srm)  python: $(python -V 2>&1)"
    echo "# libs: $(python -c 'import torch,transformers;print(f"torch {torch.__version__} transformers {transformers.__version__}")' 2>/dev/null || echo '?')"
    echo "# vllm: $(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo 'absent')"
    echo "# NOTE: a throughput number without the hardware above is not a measurement."
    printf 'utc\telapsed_s\tphase\tdetail\n'
  } > "$LOG"
  echo "  timing log started: $LOG"; sed -n '2,6p' "$LOG"
  exit 0
fi

[ -f "$T0F" ] || { echo "run './tick.sh init' first" >&2; exit 1; }
printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(( $(date -u +%s) - $(cat "$T0F") ))" "$1" "${2:-}" >> "$LOG"
tail -1 "$LOG"
