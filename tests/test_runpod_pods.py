"""Le client REST v2 : la contrainte CUDA vit sous `gpu`, les secrets ne quittent pas l'env."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[1] / "infra" / "runpod_pods.py"


@pytest.fixture
def pods() -> Any:
    spec = importlib.util.spec_from_file_location("runpod_pods", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_should_put_the_cuda_floor_under_gpu(pods: Any) -> None:
    payload = pods.build_payload(
        "p", "NVIDIA A100-SXM4-80GB", template_id="t1", cloud="COMMUNITY", min_cuda="13.0", env={"A": "1"}
    )

    assert payload["gpu"] == {"id": "NVIDIA A100-SXM4-80GB", "count": 1, "minCudaVersion": "13.0"}
    assert payload["templateId"] == "t1" and payload["cloud"] == "COMMUNITY"
    assert "disk" not in payload


def test_should_omit_the_floor_when_none_is_asked(pods: Any) -> None:
    payload = pods.build_payload("p", "g", template_id="t", cloud="SECURE", min_cuda=None, env={}, disk_gb=80)

    assert "minCudaVersion" not in payload["gpu"]
    assert payload["disk"] == 80


def test_should_copy_passthrough_variables_from_the_environment(pods: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")  # pragma: allowlist secret

    env = pods.assemble_env(["LFM2_JOB=assistant_waves", "X=a=b"], ["HF_TOKEN"])

    assert env == {"LFM2_JOB": "assistant_waves", "X": "a=b", "HF_TOKEN": "hf_x"}  # pragma: allowlist secret


def test_should_refuse_to_create_without_a_required_secret(pods: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="RUNPOD_API_KEY absent"):
        pods.assemble_env([], ["RUNPOD_API_KEY"])


def test_should_identify_itself_to_the_api_gateway(pods: Any) -> None:
    # urllib's default agent is refused by the gateway (403, code 1010).
    req = pods.build_request("POST", "/pods", {"name": "p"}, "rpa_x")

    assert req.get_header("User-agent") == pods.USER_AGENT
    assert req.get_header("Authorization") == "Bearer rpa_x"
    assert req.get_method() == "POST" and req.full_url.endswith("/v2/pods")
