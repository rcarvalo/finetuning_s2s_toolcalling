"""Create a RunPod pod through the REST v2 API, with the CUDA host floor the MCP tool lacks.

The Voxtral stack is a CUDA 13 build; on a host still on a 12.x driver the
server dies at its first CUDA call, after the whole bootstrap has been paid.
``gpu.minCudaVersion`` is a placement filter the v2 API accepts and the MCP
``create-pod`` tool does not expose — hence this thin client.

    python infra/runpod_pods.py create --name liquid-voice --gpu "NVIDIA A100-SXM4-80GB" \\
        --min-cuda 13.0 --template d5fuqge0dy --cloud COMMUNITY \\
        --env LFM2_JOB=assistant_waves --env LFM2_AUTO_DELETE=1 --pass HF_TOKEN --pass RUNPOD_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any

API = "https://api.runpod.io/v2"


def build_payload(
    name: str,
    gpu_id: str,
    *,
    template_id: str,
    cloud: str,
    min_cuda: str | None,
    env: dict[str, str],
    disk_gb: int | None = None,
) -> dict[str, Any]:
    gpu: dict[str, Any] = {"id": gpu_id, "count": 1}
    if min_cuda:
        gpu["minCudaVersion"] = min_cuda
    payload: dict[str, Any] = {"name": name, "cloud": cloud, "templateId": template_id, "gpu": gpu, "env": env}
    if disk_gb:
        payload["disk"] = disk_gb
    return payload


def assemble_env(pairs: list[str], passthrough: list[str]) -> dict[str, str]:
    """``KEY=VALUE`` literals plus variables copied from this process, never printed."""
    env = dict(pair.split("=", 1) for pair in pairs)
    for name in passthrough:
        value = os.environ.get(name, "")
        if not value:
            raise SystemExit(f"{name} absent de l'environnement — rien n'est créé")
        env[name] = value
    return env


def request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise SystemExit("RUNPOD_API_KEY absent")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return dict(json.load(response))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--gpu", required=True, help="gpu type id, ex. 'NVIDIA A100-SXM4-80GB'")
    create.add_argument("--template", required=True)
    create.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="COMMUNITY")
    create.add_argument("--min-cuda", default=None)
    create.add_argument("--disk", type=int, default=None)
    create.add_argument("--env", action="append", default=[], help="KEY=VALUE (répétable)")
    create.add_argument("--pass", dest="passthrough", action="append", default=[], help="variable copiée de l'env")
    args = parser.parse_args()

    payload = build_payload(
        args.name,
        args.gpu,
        template_id=args.template,
        cloud=args.cloud,
        min_cuda=args.min_cuda,
        env=assemble_env(args.env, args.passthrough),
        disk_gb=args.disk,
    )
    shown = {**payload, "env": {k: ("<masqué>" if k in args.passthrough else v) for k, v in payload["env"].items()}}
    print(json.dumps(shown, indent=1), file=sys.stderr)
    pod = request("POST", "/pods", payload)
    print(json.dumps({"id": pod.get("id"), "name": pod.get("name"), "gpu": pod.get("gpu"), "cloud": pod.get("cloud")}))


if __name__ == "__main__":
    main()
