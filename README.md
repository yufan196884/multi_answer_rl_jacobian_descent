This repository is forked from
[ishapuri/multi_answer_rl](https://github.com/ishapuri/multi_answer_rl).
It contains an experimental TRL implementation of:

> **Vector Policy Optimization: Training for Diversity Improves Test-Time Search**
> Ryan Bahlous-Boldi et al. ([arXiv:2605.22817](https://arxiv.org/abs/2605.22817))

This repository builds on top of [TRL](https://github.com/huggingface/trl). We thank the authors and maintainers of these projects.

---



## Installation

**1. Clone and set up the environment:**
```bash
git clone <this-repo>
cd <this-repo>
conda env create -f environment.yml
conda activate rlpa
```

**2. Install TRL at the pinned commit:**
```bash
git clone https://github.com/huggingface/trl.git
cd trl
git checkout 69ad852e5654a77f1695eb4c608906fe0c7e8624
pip install -e .
cd ..
```

**3. Log in to wandb and HuggingFace:**
```bash
wandb login
huggingface-cli login
```

**4. Configure DeepSpeed / Accelerate:**

A `deepspeed.yaml` config for 4 GPUs with ZeRO-2 is provided. If you have a different number of GPUs, update `num_processes` in `deepspeed.yaml` to match — and update `--num_processes` in your launch command accordingly.

---

## Dataset

Generate the Maze train/test JSONL files with:

```bash
python maze_dataset.py
```

This creates the paper's 1,000-example train split and 100-example test
split under `data/maze/`.

The medical dataset example is hosted on HuggingFace and loaded automatically during training:

```
mehuldamani/medDataset_25k
```

It contains 25k patient cases from DDXPlus, each with a clinical presentation and a ranked differential diagnosis. No manual download is needed.

---

## Training

### Quick Start

**RLCR (multi-answer with Brier reward):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --config_file deepspeed.yaml \
    rl_runner.py \
    --config configs/Qwen3-1-7B/rlcr_multi.yaml
```

**RLVR (multi-answer, correctness only):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --config_file deepspeed.yaml \
    rl_runner.py \
    --config configs/Qwen3-1-7B/rlvr_multi.yaml
```

**VPO (multi-answer, ranked-answer vector objectives):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --config_file deepspeed.yaml \
    rl_runner.py \
    --config configs/Qwen3-1-7B/vpo_multi_musique.yaml
```

**VPO on Maze (three routes with four reward objectives):**
```bash
python maze_dataset.py

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --config_file deepspeed.yaml \
    rl_runner.py \

    --config configs/Qwen3-1-7B/vpo_multi_maze.yaml
>>>>>>> 1d64b62740d01e6c08209d5dd70e489b3b193202
```

### Config Files


Configs live in `configs/Qwen3-1-7B/`. The provided configs are:
>>>>>>> 1d64b62740d01e6c08209d5dd70e489b3b193202

| File | Mode | Reward functions |
|------|------|-----------------|
| `rlvr_multi.yaml` | RLVR | `format` + `accuracy` |
| `rlcr_multi.yaml` | RLCR | `format` + `accuracy` + `brier` |
| `vpo_multi_maze.yaml` | Maze VPO | four-dimensional route reward + diagnostic `format` |
| `vpo_multi_musique.yaml` | MuSiQue VPO | five-dimensional support/answer reward + diagnostics |



> **Batch size note:** The effective generation batch size is:
> `num_processes × per_device_train_batch_size × gradient_accumulation_steps`
> Keep this constant or increase it when scaling to more GPUs. Lowering it may cause training instability.

### Format Patterns

`format_pattern` controls the expected output structure and which reward functions apply:

| `format_pattern` | Description | Confidence? |
|---|---|---|
| `multi_answer` | K diagnoses + K confidences + `<analysis>` | Yes |
| `multi_answer_rlvr` | K diagnoses only, no confidences | No |
| `rlcr_single_answer` | 1 diagnosis + 1 confidence | Yes |
| `rlvr_single_answer` | 1 diagnosis only | No |

### System Prompts

`sys_prompt_name` selects the instruction shown to the model. The two used in our experiments:

| `sys_prompt_name` | Use with |
|---|---|
| `og_multi_answer_short_medical` | `multi_answer` (RLCR) |
| `multi_answer_rlvr_medical` | `multi_answer_rlvr` (RLVR) |

All prompts are defined in `system_prompts.py`.

### VPO Reward Aggregation

Set `enable_vpo: true` to replace the fixed `reward_weights` policy reward with VPO aggregation. The current implementation builds a candidate reward vector from ranked gold answers: objective `j` is `1` when a generated candidate matches ranked gold answer `j`, otherwise `0`. For each prompt group, the trainer samples `vpo_num_scalarizations` symmetric Dirichlet weight vectors, applies each scalarization to every candidate, takes the best candidate in the generated set, averages those best scores, and uses that scalar as the GRPO reward.

The `reward_funcs` listed in a VPO config are still computed and logged for diagnostics, but they are not used as the policy reward while `enable_vpo` is true.


## Evaluation

### Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/rlcr_multi.json
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/rlvr_multi.json
```

### Eval Config Format

Eval configs are JSON files in `eval_configs/`. Each file is a list where the first element is shared dataset config and subsequent elements define models to evaluate:

```json
[
  {
    "dataset_name": "mehuldamani/medDataset_25k",
    "split": "test",
    "sample_size": 5000,
    "log_path": "results/my_run"
  },
  {
    "name": "my-model",
    "model": "mehuldamani/my-rlcr-checkpoint",
    "sys_prompt_name": "og_multi_answer_short_medical",
    "check_fn": "confidence_verifier",
    "vllm_task": ["confidence_at_end", "ans_at_end"],
    "num_candidates": 3,
    "temperature": 0.7,
    "max_tokens": 1536,
    "check_fn_args": {"format_pattern": "multi_answer"}
  }
]
```

Replace `"model"` with your local checkpoint path or HuggingFace model ID.

---

## Inference

To run a single example through a trained model:

```bash
python inference_example.py
```

Edit `inference_example.py` to change the model, system prompt, or input case.

---
