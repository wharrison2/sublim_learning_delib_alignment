# Pod provisioning

`pod.py` — create, inspect, and **terminate** RunPod pods via the REST API.

```bash
runpodctl config --apiKey <key>          # once; writes ~/.runpod/config.toml

python pod.py up --volume <vol_id>       # prints POD_ID
python pod.py ssh  <pod_id>              # prints the ssh command
python pod.py down <pod_id>              # ← billing runs until you do this
```

The key is read from `~/.runpod/config.toml` or `$RUNPOD_API_KEY` — never passed as an
argument, so it stays out of shell history.

## Why a teardown command exists at all

RunPod bills wall-clock while the pod **exists**, not while the GPU is busy. A pod left
running overnight costs more than every compute estimate in `timing_notes.md` combined. The
web console makes that easy to do; `pod.py down` is the antidote. Run it even after a failed
run — especially after a failed run.

## GPU selection is discovered, not hardcoded

`up` lists GPU types, filters to `--min-vram` (default 80 GB), sorts by price, and passes
the cheapest four with `gpuTypePriority: "availability"` so RunPod takes whichever is free.
Hardcoded type IDs go stale, and a stale ID fails at provision time.

## Session A: prompt generation

```bash
python pod.py up --volume <vol_id> --n 2000
python pod.py ssh <pod_id>
# on the pod:
git clone https://github.com/wharrison2/sublim_learning_delib_alignment.git
cd sublim_learning_delib_alignment/prompts && ./run_on_pod.sh
# back home:
python pod.py down <pod_id>
```

**Attach the volume even though prompt generation doesn't need one.** Without it the output
dies with the pod and you have to exfiltrate `gen_prompts.jsonl` before teardown. With it the
file lands on `/workspace` and Session B picks it up. The cost is Secure Cloud rates
(a network volume forces Secure) — about $0.17 more across a 35-minute run.

**`HF_HOME` is set to `/root/hf`, on the container disk — not the volume.** Mistral is 48 GB
and would otherwise fill a 50 GB volume. Only the small output belongs on `/workspace`.

## Why SSH rather than full automation, for this run

The API takes `dockerStartCmd`, so a pod can run the job and terminate itself, and that is
the right shape for G8's ~30 training runs. It is the wrong shape here: prompt generation
produces something whose *quality* has to be eyeballed, and if the prompts come back bland
you want to see that while the pod is still warm rather than after teardown. Script the
provisioning, watch the run.

## Untested

`pod.py` has not been run against a live account — there was no API key available when it
was written. Two things to expect on first use: the `/gputypes` response shape (the script
prints candidates before creating anything, so a mismatch is visible immediately and creates
no pod), and the SSH port mapping field names in the pod object. Run `up` and read the echoed
JSON before trusting `ssh`.
