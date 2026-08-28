# Pod provisioning

`pod.py` — create, inspect, and **terminate** RunPod pods via the REST API.

## Where each command runs

Three different machines. Everything in this file except the `run_on_pod.sh` block runs on
**your own computer**.

| | where | what |
|---|---|---|
| **Browser** | RunPod console | **Once only:** register your SSH public key (Settings → SSH Public Keys). Nothing else needs the console |
| **Your Mac** | local terminal | every `pod.py` command, and the `ssh` command it prints |
| **The pod** | after ssh connects you | `git clone`, `./run_on_pod.sh` |

## Setup (once)

**1. SSH key → RunPod console.** `pod.py ssh` prints an `ssh` command, but the pod only
accepts it if RunPod holds your public key. Paste `~/.ssh/id_ed25519.pub` into
Settings → SSH Public Keys. This is the only browser step.

**2. API key on disk.** No `runpodctl` install needed:

```bash
mkdir -p ~/.runpod && chmod 700 ~/.runpod
printf %s 'rpa_...' > ~/.runpod/key && chmod 600 ~/.runpod/key
```

Read from `$RUNPOD_API_KEY`, then `~/.runpod/key`, then `~/.runpod/config.toml` (runpodctl's
own file, if you happen to have it). **Never passed as an argument** — that lands in shell
history.

## Run (on your Mac)

```bash
python pod.py vol-list                   # find the volume id
python pod.py up --volume <vol_id>       # prints POD_ID
python pod.py ssh  <pod_id>              # prints an ssh command -- run it
python pod.py down <pod_id>              # ← billing runs until you do this
```

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
