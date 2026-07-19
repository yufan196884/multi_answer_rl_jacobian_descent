#!/usr/bin/env python3
"""Evaluate a multi-answer MuSiQue checkpoint with best@k metrics."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from dataset_processing import make_musique_generation_dataset
from reward_fns import (
    format_reward,
    musique_candidate_reward_vectors,
)
from system_prompts import get_sys_prompt_formatted


REWARD_NAMES = (
    "hop_1",
    "hop_2",
    "hop_3",
    "hop_4",
    "answer_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a VPO MuSiQue checkpoint."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Local checkpoint directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path. Defaults to --model.",
    )

    parser.add_argument(
        "--dataset-name",
        default="bdsaglam/musique",
    )
    parser.add_argument(
        "--dataset-config",
        default="answerable",
    )
    parser.add_argument(
        "--split",
        default="validation",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--selection",
        choices=("first", "balanced"),
        default="balanced",
        help=(
            "'first' matches the current trainer's first-300 selection; "
            "'balanced' selects equally across 2-, 3-, and 4-hop questions."
        ),
    )

    parser.add_argument(
        "--candidates-per-chain",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[3, 5, 10, 30],
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1536,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/musique"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debugging limit applied after subset selection.",
    )

    return parser.parse_args()


def select_balanced_hops(
    dataset: Dataset,
    sample_size: int,
    seed: int,
) -> Dataset:
    """
    Select approximately equal numbers of 2-, 3-, and 4-hop questions.

    The exact 300 IDs used in the paper are not present in this repository,
    so the selected IDs should be saved and reused for all checkpoints.
    """
    hop_groups: dict[int, list[int]] = {
        2: [],
        3: [],
        4: [],
    }

    for index, decomposition in enumerate(
        dataset["question_decomposition"]
    ):
        hop_count = len(decomposition or [])

        if hop_count in hop_groups:
            hop_groups[hop_count].append(index)

    rng = random.Random(seed)

    for indices in hop_groups.values():
        rng.shuffle(indices)

    base_quota = sample_size // len(hop_groups)
    remainder = sample_size % len(hop_groups)

    selected_indices: list[int] = []

    for offset, hop_count in enumerate(sorted(hop_groups)):
        quota = base_quota + (1 if offset < remainder else 0)
        available = hop_groups[hop_count]

        if len(available) < quota:
            raise ValueError(
                f"Requested {quota} examples for {hop_count}-hop "
                f"questions, but only {len(available)} exist."
            )

        selected_indices.extend(available[:quota])

    # Remove any systematic ordering by hop count.
    rng.shuffle(selected_indices)

    return dataset.select(selected_indices)


def musique_scalar_reward(
    reward_vector: Sequence[float],
) -> float:
    """GRPO/final-evaluation scalar used for MuSiQue."""
    if len(reward_vector) != 5:
        raise ValueError(
            "MuSiQue reward vectors must have five components."
        )

    hop_total = sum(float(value) for value in reward_vector[:4])
    answer_f1 = float(reward_vector[4])

    return (hop_total + 3.0 * answer_f1) / 7.0


def pairwise_l1_diversity(
    reward_vectors: Sequence[Sequence[float]],
) -> float:
    rewards = np.asarray(
        reward_vectors,
        dtype=np.float64,
    )

    if len(rewards) < 2:
        return 0.0

    pairwise = np.abs(
        rewards[:, None, :] - rewards[None, :, :]
    ).sum(axis=-1)

    rows, columns = np.triu_indices(
        len(rewards),
        k=1,
    )

    return float(pairwise[rows, columns].mean())


def score_completion_chains(
    row: dict,
    texts: list[str],
    candidates_per_chain: int,
) -> tuple[list[list[list[float]]], list[float]]:
    """
    Return:
      vectors_by_chain[chain][candidate][reward_dimension]
      strict_format_scores[chain]

    Metadata is explicitly repeated once per completion. This is important for
    aliases and nested paragraph/decomposition fields.
    """
    completion_objects = [
        [
            {
                "role": "assistant",
                "content": text,
            }
        ]
        for text in texts
    ]

    chain_count = len(completion_objects)

    strict_format_scores = format_reward(
        format_pattern="musique_multi_answer",
        completions=completion_objects,
        num_candidates=candidates_per_chain,
    )

    vectors_by_chain = musique_candidate_reward_vectors(
        format_pattern="musique_multi_answer",
        completions=completion_objects,
        answer=[
            row.get("answer", "")
            for _ in range(chain_count)
        ],
        answer_aliases=[
            row.get("answer_aliases", [])
            for _ in range(chain_count)
        ],
        paragraphs=[
            row.get("paragraphs", [])
            for _ in range(chain_count)
        ],
        question_decomposition=[
            row.get("question_decomposition", [])
            for _ in range(chain_count)
        ],
        num_candidates=candidates_per_chain,

        # Match the repository's format requirements.
        vpo_apply_format_gate=True,

        # Do not erase duplicate candidates at evaluation time. Duplicates
        # should instead naturally reduce diversity and best@k headroom.
        vpo_apply_uniqueness_gate=False,
    )

    return vectors_by_chain, strict_format_scores


def main() -> None:
    args = parse_args()

    k_values = sorted(set(args.k_values))

    if not k_values or min(k_values) <= 0:
        raise ValueError("All k-values must be positive.")

    max_k = max(k_values)

    chains_per_question = math.ceil(
        max_k / args.candidates_per_chain
    )

    raw_dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
    )

    if args.selection == "first":
        dataset = raw_dataset.select(
            range(min(args.sample_size, len(raw_dataset)))
        )
    else:
        dataset = select_balanced_hops(
            raw_dataset,
            sample_size=args.sample_size,
            seed=args.seed,
        )

    if args.limit is not None:
        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

    system_prompt = get_sys_prompt_formatted(
        "musique_multi_answer",
        args.candidates_per_chain,
    )

    processed_dataset = make_musique_generation_dataset(
        dataset,
        system_prompt,
    )

    tokenizer_source = args.tokenizer or args.model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )

    prompts = [
        tokenizer.apply_chat_template(
            processed_dataset[index]["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for index in range(len(processed_dataset))
    ]

    sampling_params = SamplingParams(
        n=chains_per_question,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    request_outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        args.output_dir / "predictions.jsonl"
    )
    metrics_path = args.output_dir / "metrics.json"
    selected_ids_path = (
        args.output_dir / "selected_ids.json"
    )

    selected_ids = [
        processed_dataset[index].get("id", index)
        for index in range(len(processed_dataset))
    ]

    selected_ids_path.write_text(
        json.dumps(selected_ids, indent=2),
        encoding="utf-8",
    )

    best_values: dict[int, list[float]] = {
        k: [] for k in k_values
    }

    answer_f1_at_max_k: list[float] = []
    diversities: list[float] = []
    format_scores: list[float] = []
    all_vectors: list[list[float]] = []

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as predictions_file:
        for row, request_output in zip(
            processed_dataset,
            request_outputs,
            strict=True,
        ):
            ordered_outputs = sorted(
                request_output.outputs,
                key=lambda output: output.index,
            )

            completion_texts = [
                output.text
                for output in ordered_outputs
            ]

            vectors_by_chain, chain_format_scores = (
                score_completion_chains(
                    row=row,
                    texts=completion_texts,
                    candidates_per_chain=args.candidates_per_chain,
                )
            )

            # The paper concatenates chains in draw order.
            candidate_vectors = [
                vector
                for chain_vectors in vectors_by_chain
                for vector in chain_vectors
            ][:max_k]

            scalar_rewards = [
                musique_scalar_reward(vector)
                for vector in candidate_vectors
            ]

            question_best_at_k = {}

            for k in k_values:
                value = float(max(scalar_rewards[:k]))
                best_values[k].append(value)
                question_best_at_k[str(k)] = value

            f1_value = float(
                max(vector[4] for vector in candidate_vectors)
            )

            diversity = pairwise_l1_diversity(
                candidate_vectors
            )

            answer_f1_at_max_k.append(f1_value)
            diversities.append(diversity)
            format_scores.extend(chain_format_scores)
            all_vectors.extend(candidate_vectors)

            record = {
                "id": row.get("id"),
                "question": row.get("question"),
                "hop_count": len(
                    row.get("question_decomposition", [])
                ),
                "best_at_k": question_best_at_k,
                f"f1@{max_k}": f1_value,
                f"diversity@{max_k}": diversity,
                "scalar_rewards": scalar_rewards,
                "reward_vectors": candidate_vectors,
                "chains": [
                    {
                        "text": text,
                        "strict_format_valid": bool(format_score),
                        "reward_vectors": vectors,
                    }
                    for text, format_score, vectors in zip(
                        completion_texts,
                        chain_format_scores,
                        vectors_by_chain,
                        strict=True,
                    )
                ],
            }

            predictions_file.write(
                json.dumps(record) + "\n"
            )

    rewards_array = np.asarray(
        all_vectors,
        dtype=np.float64,
    )

    metrics = {
        "num_questions": len(processed_dataset),
        "selection": args.selection,
        "candidates_per_chain": args.candidates_per_chain,
        "chains_per_question": chains_per_question,
        "candidate_pool_size": max_k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        **{
            f"best@{k}": float(np.mean(values))
            for k, values in best_values.items()
        },
        f"f1@{max_k}": float(
            np.mean(answer_f1_at_max_k)
        ),
        f"diversity@{max_k}": float(
            np.mean(diversities)
        ),
        "strict_format_rate": float(
            np.mean(format_scores)
        ),
        "mean_reward_components": {
            reward_name: float(
                rewards_array[:, dimension].mean()
            )
            for dimension, reward_name
            in enumerate(REWARD_NAMES)
        },
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Metrics:      {metrics_path}")
    print(f"Predictions:  {predictions_path}")
    print(f"Selected IDs: {selected_ids_path}")


if __name__ == "__main__":
    main()