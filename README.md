# Qwen3.8 vLLM Launchers

Shell launchers and a benchmark command for serving Qwen3.8 27B NVFP4 model variants with vLLM.

## Prerequisites

- Linux with an NVIDIA driver and CUDA-compatible GPU software stack.
- Two GPUs are expected by the launchers (`--tensor-parallel-size 2`). The exact driver, GPU, and CUDA requirements depend on the PyTorch and vLLM builds installed on the host.
- Bash and Python 3.13.
- A CUDA-compatible installation of PyTorch, vLLM, and FlashInfer in the project virtual environment. The required wheel/index source depends on the host's CUDA stack; the setup script intentionally does not guess or install these packages.
- Hugging Face CLI installed and configured, and access to the selected model repositories. Complete any required model terms/access approval before launching.

The current machine's environment reports CPython 3.13, vLLM 0.30.0, PyTorch 2.13.0+cu132, Transformers 5.17.0, Hugging Face Hub 1.33.0, and FlashInfer 0.6.18.post1. These are observations from one working environment, not a complete or portable installation recipe. Install versions and CUDA builds that are compatible with the target host and each other.

## Environment

Create the project virtual environment:

```bash
./setup_env.sh
source unsloth-nvfp4-env/bin/activate
```

`setup_env.sh` creates `unsloth-nvfp4-env` with Python 3.13. If a usable Python 3.13 environment already exists there, it leaves it unchanged. It does not install Python packages or configure Hugging Face.

Install the host-compatible PyTorch, vLLM, and FlashInfer builds into the environment using the installation source appropriate for the machine. The launch scripts expect the `vllm` command to be available after activation.

## Hugging Face Access

Hugging Face setup is a prerequisite and is not performed by the environment script. Install the Hugging Face CLI and authenticate outside this repository when required:

```bash
hf auth login
hf auth whoami
```

Do not put a Hugging Face token in this repository. The default Hugging Face cache is outside the project directory; ensure the selected model is accessible and cached or that the host can download it.

The local vLLM API key is separate from Hugging Face authentication. Put the desired API key in the root-level `API_KEY.txt`; all three launchers read it from there. This file is ignored by Git.

Generate a 256-bit key in the project root without overwriting an existing key file:

```bash
( umask 077 && set -o noclobber && openssl rand -hex 32 > API_KEY.txt )
```

This creates a 64-character hexadecimal key with owner-only file permissions. If `API_KEY.txt` already exists, the command fails rather than replacing it.

## Launchers

Run one launcher at a time; each stops existing processes matching `vllm serve` before starting its server. The launchers use vLLM's default listen address and port (usually `localhost:8000`).

| Script | Model source |
| --- | --- |
| `start_qwen_gittensor.sh` | Local Hugging Face cache snapshot at a machine-specific path under `/data/huggingface/hub/`. Update the path in the script if the snapshot is stored elsewhere. |
| `start_qwen_quasar.sh` | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` from Hugging Face. |
| `start_qwen_unsloth.sh` | `unsloth/Qwen3.8-27B-NVFP4` from Hugging Face. |

Each script includes its own context length, cache, and speculative decoding settings. Review those options against available GPU memory before changing them.

## Benchmark

With a server running, `./test_qwen.sh` runs `vllm bench serve` against `localhost:8000`. It requires `API_KEY.txt` and currently points its tokenizer at the Gittensor snapshot path used by `start_qwen_gittensor.sh`; update that path if needed. The benchmark's requested model name should match the model name advertised by the running server if the endpoint validates model names.

## Resources

See [resources.md](resources.md) for the original project and model comparison links.