#!/usr/bin/env python3
"""Provision / inspect / terminate a RunPod pod for this project.

    python pod.py up      --volume <vol_id> [--secure] [--disk 70]
    python pod.py status  <pod_id>
    python pod.py ssh     <pod_id>          # prints the ssh command
    python pod.py down    <pod_id>

WHY THIS EXISTS: RunPod bills wall-clock while the pod *exists*, not while the GPU is busy.
The web console's failure mode is a pod left running overnight, which would dwarf every
compute estimate in timing_notes.md. Always have a teardown path.

CREDENTIALS: reads the key from ~/.runpod/config.toml (written by `runpodctl config
--apiKey ...`) or $RUNPOD_API_KEY. Never pass it on the command line -- it lands in shell
history.

GPU SELECTION is discovered, not hardcoded: the script lists GPU types and picks an
available one with >= --min-vram GB. Hardcoded type IDs go stale.
"""
import argparse, json, os, pathlib, re, sys, time, urllib.error, urllib.request

API = "https://rest.runpod.io/v1"


def key() -> str:
    """API key, in preference order. Never taken from argv -- that lands in shell history."""
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    # Plain key file. Simplest path: no Go toolchain to install for one string.
    plain = pathlib.Path.home() / ".runpod" / "key"
    if plain.exists():
        return plain.read_text().strip()
    # runpodctl's own config, if it happens to be installed.
    cfg = pathlib.Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        m = re.search(r'apiKey\s*=\s*"([^"]+)"', cfg.read_text())
        if m:
            return m.group(1)
    sys.exit(
        "No API key found. Pick one:\n"
        "  mkdir -p ~/.runpod && chmod 700 ~/.runpod\n"
        "  printf %s 'rpa_...' > ~/.runpod/key && chmod 600 ~/.runpod/key\n"
        "or:  export RUNPOD_API_KEY=rpa_...\n"
        "or:  runpodctl config --apiKey rpa_...   (if runpodctl is installed)")


def api(method, path, body=None):
    req = urllib.request.Request(
        f"{API}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> {e.code}\n{e.read().decode()[:800]}")


GQL = "https://api.runpod.io/graphql"


GQL = "https://api.runpod.io/graphql"


def gpu_stock(dc=None, min_vram=0, allow_amd=False):
    """GPU types with live stock, optionally scoped to one datacenter.

    NOTE: REST v1 has no GPU-types endpoint -- /v1/gputypes and every variant 400 with
    "path does not exist in the specification". This lives only on the older GraphQL API,
    which answers unauthenticated but 403s the default Python-urllib User-Agent.

    Passing dataCenterId makes lowestPrice return null for cards that DC cannot serve, so
    filtering on a non-null price is what prevents "no instances currently available".
    """
    arg = "gpuCount:1" + (f',dataCenterId:"{dc}"' if dc else "")
    q = ("query{ gpuTypes { id memoryInGb secureCloud "
         f"lowestPrice(input:{{{arg}}}){{ uninterruptablePrice stockStatus }} }} }}")
    req = urllib.request.Request(
        GQL, method="POST", data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.4.0"})
    with urllib.request.urlopen(req) as r:
        types = (json.load(r).get("data") or {}).get("gpuTypes") or []

    def price(g):
        return (g.get("lowestPrice") or {}).get("uninterruptablePrice")

    out = [g for g in types
           if (g.get("memoryInGb") or 0) >= min_vram
           and price(g) is not None          # null price == not served here
           # AMD needs the ROCm vLLM build; a different stack is not worth the risk here.
           and (allow_amd or "NVIDIA" in g["id"])]
    out.sort(key=price)
    return out


def gpus(a):
    g = gpu_stock(a.datacenter, a.min_vram, a.allow_amd)
    where = a.datacenter or "anywhere"
    if not g:
        print(f"  no GPU >= {a.min_vram}GB with stock in {where}")
        return
    print(f"{'id':46}{'VRAM':>6}{'$/hr':>8}  stock in {where}")
    for x in g:
        lp = x.get("lowestPrice") or {}
        print(f"  {x['id'][:44]:44}{x.get('memoryInGb'):>6}"
              f"{lp.get('uninterruptablePrice'):>8}  {lp.get('stockStatus')}")


def pick_gpu(dc, min_vram, allow_amd=False, limit=12):
    try:
        ok = gpu_stock(dc, min_vram, allow_amd)
    except Exception as e:
        sys.exit(f"GPU discovery failed ({e}). Pass --gpu '<exact id>' to skip it.")
    if not ok:
        sys.exit(f"No GPU >= {min_vram}GB with stock in {dc or 'any DC'}.\n"
                 f"Check with:  python pod.py gpus --datacenter {dc or 'US-KS-2'}\n"
                 "A network volume pins you to its datacenter -- if that DC has nothing, "
                 "you either wait, or recreate the volume elsewhere.")
    print(f"  GPU candidates with stock in {dc or 'any DC'} (cheapest first):", file=sys.stderr)
    for g in ok[:limit]:
        lp = g.get("lowestPrice") or {}
        print(f"    {g['id']:46} {g.get('memoryInGb')}GB  "
              f"${lp.get('uninterruptablePrice')}/hr  {lp.get('stockStatus')}", file=sys.stderr)
    # Pass ALL of them: gpuTypePriority "availability" lets RunPod take whichever is free,
    # and truncating to the cheapest few is how "no instances available" happens.
    return [g["id"] for g in ok[:limit]]


def up(a):
    body = {
        "name": a.name,
        "imageName": a.image,
        "gpuCount": 1,
        "gpuTypeIds": [a.gpu] if a.gpu else pick_gpu(a.datacenter, a.min_vram, a.allow_amd),
        "gpuTypePriority": "availability",
        "containerDiskInGb": a.disk,
        "cloudType": "SECURE" if (a.secure or a.volume) else "COMMUNITY",
        "ports": ["22/tcp"],
        # HF_HOME on the CONTAINER disk, not the volume: Mistral is 48GB and would
        # otherwise fill a 50GB volume. Only the small output belongs on /workspace.
        "env": {"HF_HOME": "/root/hf", "N": str(a.n)},
    }
    if a.volume:
        body["networkVolumeId"] = a.volume
        body["volumeMountPath"] = "/workspace"
    else:
        body["dataCenterIds"] = [a.datacenter]
        body["dataCenterPriority"] = "availability"

    print(f"creating pod (cloud={body['cloudType']}, disk={a.disk}GB, "
          f"volume={a.volume or 'none'})...", file=sys.stderr)
    pod = api("POST", "/pods", body)
    pid = pod.get("id") or pod.get("podId")
    print(json.dumps(pod, indent=1)[:600], file=sys.stderr)
    print(f"\nPOD_ID={pid}")
    print(f"\n  watch:      python pod.py status {pid}"
          f"\n  connect:    python pod.py ssh {pid}"
          f"\n  TEAR DOWN:  python pod.py down {pid}    <-- billing runs until you do", file=sys.stderr)
    return pid


def vol_list(a):
    v = api("GET", "/networkvolumes")
    v = v if isinstance(v, list) else v.get("data", [])
    if not v:
        print("(no network volumes)")
    for x in v:
        print(f"  {x.get('id'):24} {x.get('name'):22} {x.get('size')}GB  "
              f"{x.get('dataCenterId')}")


def vol_new(a):
    body = {"name": a.name, "size": a.size, "dataCenterId": a.datacenter}
    print(f"POST /networkvolumes {json.dumps(body)}", file=sys.stderr)
    v = api("POST", "/networkvolumes", body)
    print(json.dumps(v, indent=1))


def vol_rm(a):
    api("DELETE", f"/networkvolumes/{a.volume_id}")
    print(f"deleted {a.volume_id}")


def vol_probe(a):
    """Which datacenters will actually accept this volume right now?

    A 500 on create is a SERVER error -- the request was fine. Usually it means either a
    transient fault or storage capacity exhausted in that datacenter, surfacing as a 500
    rather than a clean 4xx. This isolates which: same request, several datacenters.

    Creates and IMMEDIATELY deletes. Anything it fails to clean up is printed loudly --
    a stray volume bills continuously.
    """
    dcs = a.datacenters.split(",")
    made = []
    print(f"probing {a.size}GB in: {', '.join(dcs)}\n", file=sys.stderr)
    for dc in dcs:
        body = {"name": f"{a.name}-probe", "size": a.size, "dataCenterId": dc}
        req = urllib.request.Request(
            f"{API}/networkvolumes", method="POST", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                v = json.loads(r.read() or "{}")
            vid = v.get("id")
            made.append((dc, vid))
            print(f"  {dc:12} OK    -> {vid}")
        except urllib.error.HTTPError as e:
            print(f"  {dc:12} {e.code}   {e.read().decode()[:160]}")
        except Exception as e:
            print(f"  {dc:12} ERR   {e}")
    for dc, vid in made:
        if not vid:
            continue
        try:
            api("DELETE", f"/networkvolumes/{vid}")
            print(f"  cleaned up {vid} ({dc})", file=sys.stderr)
        except SystemExit:
            print(f"\n  ⚠⚠ COULD NOT DELETE {vid} in {dc} -- delete it by hand, "
                  f"it is billing now", file=sys.stderr)


def status(a):
    p = api("GET", f"/pods/{a.pod_id}")
    print(json.dumps(p, indent=1)[:2000])


def ssh(a):
    p = api("GET", f"/pods/{a.pod_id}")
    ip = p.get("publicIp")
    port = next((x.get("publicPort") for x in (p.get("portMappings") or [])
                 if str(x.get("privatePort")) == "22"), None)
    if not (ip and port):
        sys.exit(f"No public SSH mapping yet (status={p.get('desiredStatus')}). "
                 "Wait for the pod to finish starting and retry.")
    print(f"ssh root@{ip} -p {port} -i ~/.ssh/id_ed25519")
    print("\nthen, on the pod:", file=sys.stderr)
    print("  git clone https://github.com/wharrison2/sublim_learning_delib_alignment.git \\\n"
          "    && cd sublim_learning_delib_alignment/prompts && ./run_on_pod.sh", file=sys.stderr)


def down(a):
    api("DELETE", f"/pods/{a.pod_id}")
    print(f"terminated {a.pod_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up"); u.set_defaults(fn=up)
    u.add_argument("--volume", default=None, help="network volume id (forces Secure Cloud)")
    u.add_argument("--datacenter", default="US-CA-2")
    u.add_argument("--disk", type=int, default=70, help="container disk GB; Mistral needs ~60")
    u.add_argument("--min-vram", type=int, default=80)
    u.add_argument("--gpu", default=None,
                   help="Exact GPU type id, skipping discovery, e.g. 'NVIDIA A100 80GB PCIe'")
    u.add_argument("--allow-amd", action="store_true",
                   help="Include AMD cards. Off by default: vLLM on ROCm is a different "
                        "stack and not worth the risk on a run this small.")
    u.add_argument("--n", type=int, default=2000, help="prompt target, passed as $N")
    u.add_argument("--secure", action="store_true")
    u.add_argument("--name", default="prompt-gen")
    u.add_argument("--image",
                   default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04")
    for name, fn in (("status", status), ("ssh", ssh), ("down", down)):
        s = sub.add_parser(name); s.set_defaults(fn=fn); s.add_argument("pod_id")

    gp = sub.add_parser("gpus"); gp.set_defaults(fn=gpus)
    gp.add_argument("--datacenter", default=None, help="e.g. US-CA-2; omit for global")
    gp.add_argument("--min-vram", type=int, default=80)
    gp.add_argument("--allow-amd", action="store_true")

    sub.add_parser("vol-list").set_defaults(fn=vol_list)
    vn = sub.add_parser("vol-new"); vn.set_defaults(fn=vol_new)
    vn.add_argument("--name", default="subliminal-em")
    vn.add_argument("--size", type=int, default=50)
    vn.add_argument("--datacenter", default="US-CA-2")
    vr = sub.add_parser("vol-rm"); vr.set_defaults(fn=vol_rm)
    vr.add_argument("volume_id")
    vp = sub.add_parser("vol-probe"); vp.set_defaults(fn=vol_probe)
    vp.add_argument("--name", default="subliminal-em")
    vp.add_argument("--size", type=int, default=50)
    vp.add_argument("--datacenters",
                    default="US-CA-2,US-KS-2,US-GA-1,EU-RO-1,CA-MTL-1")
    a = ap.parse_args()
    a.fn(a)
