# Running on Colab from the terminal

Two paths reach Colab from this repo. Only one of them can be automated.

| Path | Provision a GPU | Run code | Verdict |
|---|---|---|---|
| `google-colab-cli` (`colab` command) | ✅ `--gpu L4` | ✅ local script on the remote VM | **use this** |
| Colab MCP browser bridge | ❌ no runtime tool | ✅ cell-by-cell in an open notebook | interactive editing only |

The MCP bridge exposes eight cell-manipulation tools (`add_code_cell`,
`run_code_cell`, `get_cells`, …) that act on a notebook already open in the
user's browser. It has **no** way to create a runtime or choose an accelerator —
that stays a manual UI action. The CLI has both.

## Verified on 2026-08-22

```
colab new -s <name> --gpu L4     →  NVIDIA L4, 23034 MiB, driver 580.82.07
runtime python                   →  3.13.15
```

The pinned stack resolves for that interpreter, including the versions the
vLLM-Omni plugin was written against:

| Package | Needed | Available on the L4 |
|---|---|---|
| `vllm` | 0.22.x | 0.22.0, 0.22.1 (up to 0.27.1) |
| `vllm-omni` | 0.22.0 | 0.22.0 (up to 0.26.0) |
| `liquid-audio` | ≥1.3.0 | 1.3.0 |
| `torch` / `transformers` / `onnxruntime` | — | 2.13.0 / 5.15.1 / 1.29.0 |

**Pin explicitly.** `pip install vllm` resolves to 0.27 — the plugin's documented
workarounds (`_set_final_only_for_llm_stages`, `enable_prefix_caching=False`,
`async_scheduling=False`) target 0.22 and are not known to hold on 0.26+.

## Operating procedure

```bash
colab new -s baseline --gpu L4          # provision (L4 | T4 | A100 | H100 | G4)
colab status -s baseline                # confirm the VM answers
colab install -s baseline <packages>    # uses uv on the VM
colab exec -s baseline --timeout 900 -f scripts/run_eval.py
colab download -s baseline /content/report.json ./reports/
colab log -s baseline -o session.ipynb  # capture the run as a notebook artifact
colab stop -s baseline                  # ALWAYS: idle VMs burn compute units
```

### Rules for agents

- **`--timeout` defaults to 10 seconds.** Anything that installs, downloads a
  checkpoint or runs a campaign needs an explicit, generous value. The default
  produces a `ReadTimeout` that looks like a crash but is only the client giving up.
- **Never run `colab repl` or `colab console` interactively.** They expect a TTY
  and will hang an agent. Pipe stdin instead: `echo "cmd" | colab console -s <name>`.
- **Always `colab stop`** when a run finishes, including on failure.
- `colab sessions` lists everything running — check it before creating a second
  L4, and do not stop a session you did not create.
- Session state lives in `~/.config/colab-cli/sessions.json`; never edit it by hand.

`colab skill` prints the bundled operator guide, and `colab help` lists every
command.
