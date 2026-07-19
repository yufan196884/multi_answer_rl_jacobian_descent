#!/usr/bin/env python3
"""Evaluate a multi-answer Maze checkpoint using best@k and reward diversity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from maze_dataset import (
    extract_numbered_routes,
    maze_from_record,
    simulate_route,
    uniform_scalar_reward,
)
from system_prompts import get_sys_prompt_formatted


REWARD_NAMES = (
    "completion",
    "gold",
    "diamond",
    "lava_avoidance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a VPO Maze checkpoint."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Local checkpoint directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer path or Hugging Face model ID. "
            "Defaults to --model."
        ),
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/maze/test.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/maze"),
    )

    parser.add_argument(
        "--routes-per-rollout",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[3, 5, 10, 30],
    )

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

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
        "--limit",
        type=int,
        default=None,
        help="Optional debugging limit.",
    )

    return parser.parse_args()


def make_messages(
    question: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Match the training-time conversational dataset structure."""
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"\n\nPROBLEM: {question}\n\n",
        },
    ]


def pairwise_l1_diversity(
    reward_vectors: Sequence[Sequence[float]],
) -> float:
    """Average L1 distance over all unordered pairs."""
    rewards = np.asarray(reward_vectors, dtype=np.float64)

    if rewards.shape[0] < 2:
        return 0.0

    distances = np.abs(
        rewards[:, None, :] - rewards[None, :, :]
    ).sum(axis=-1)

    row_indices, col_indices = np.triu_indices(
        rewards.shape[0],
        k=1,
    )

    return float(distances[row_indices, col_indices].mean())


def score_rollout(
    maze,
    completion_text: str,
    routes_per_rollout: int,
) -> tuple[list[list[float]], list[str | None], bool]:
    """
    Parse and score one multi-route completion.

    Invalidly formatted completions contribute the expected number of zero
    vectors. They must not be skipped, because skipping would artificially
    improve best@k.
    """
    routes = extract_numbered_routes(
        completion_text,
        num_routes=routes_per_rollout,
    )

    if routes is None:
        zero_vectors = [
            [0.0, 0.0, 0.0, 0.0]
            for _ in range(routes_per_rollout)
        ]
        return zero_vectors, [None] * routes_per_rollout, False

    reward_vectors = [
        list(simulate_route(maze, route))
        for route in routes
    ]
    route_strings = [
        " ".join(route)
        for route in routes
    ]

    return reward_vectors, route_strings, True


def main() -> None:
    args = parse_args()

    if not args.test_file.exists():
        raise FileNotFoundError(
            f"Maze test file does not exist: {args.test_file}"
        )

    k_values = sorted(set(args.k_values))

    if not k_values or min(k_values) <= 0:
        raise ValueError("All k-values must be positive.")

    max_k = max(k_values)

    # Each model completion contains three routes.
    rollouts_per_maze = math.ceil(
        max_k / args.routes_per_rollout
    )
    candidates_per_maze = (
        rollouts_per_maze * args.routes_per_rollout
    )

    dataset = load_dataset(
        "json",
        data_files={"test": str(args.test_file)},
        split="test",
    )

    if args.limit is not None:
        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

    if len(dataset) != 100 and args.limit is None:
        print(
            f"Warning: expected 100 test mazes, found {len(dataset)}."
        )

    tokenizer_source = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )

    system_prompt = get_sys_prompt_formatted(
        "maze_multi_answer",
        args.routes_per_rollout,
    )

    prompts = [
        tokenizer.apply_chat_template(
            make_messages(row["question"], system_prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in dataset
    ]

    sampling_params = SamplingParams(
        n=rollouts_per_maze,
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

    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"

    per_k_values: dict[int, list[float]] = {
        k: [] for k in k_values
    }
    per_maze_diversities: list[float] = []
    all_reward_vectors: list[list[float]] = []
    all_format_validities: list[float] = []

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as predictions_file:
        for row, request_output in zip(
            dataset,
            request_outputs,
            strict=True,
        ):
            maze = maze_from_record(row)

            maze_reward_vectors: list[list[float]] = []
            rollout_records = []

            # Preserve vLLM's candidate-generation order.
            generated_rollouts = sorted(
                request_output.outputs,
                key=lambda output: output.index,
            )

            for rollout_index, output in enumerate(
                generated_rollouts
            ):
                reward_vectors, routes, format_valid = score_rollout(
                    maze=maze,
                    completion_text=output.text,
                    routes_per_rollout=args.routes_per_rollout,
                )

                maze_reward_vectors.extend(reward_vectors)
                all_format_validities.append(
                    1.0 if format_valid else 0.0
                )

                rollout_records.append(
                    {
                        "rollout_index": rollout_index,
                        "text": output.text,
                        "format_valid": format_valid,
                        "routes": routes,
                        "reward_vectors": reward_vectors,
                    }
                )

            # Only the first max_k candidates belong to the reported pool.
            evaluation_vectors = maze_reward_vectors[:max_k]
            scalar_rewards = [
                uniform_scalar_reward(vector)
                for vector in evaluation_vectors
            ]

            best_at_k = {}

            for k in k_values:
                value = float(max(scalar_rewards[:k]))
                best_at_k[str(k)] = value
                per_k_values[k].append(value)

            diversity = pairwise_l1_diversity(
                evaluation_vectors
            )

            per_maze_diversities.append(diversity)
            all_reward_vectors.extend(evaluation_vectors)

            prediction_record = {
                "id": row.get("id"),
                "seed": row["seed"],
                "best_at_k": best_at_k,
                "diversity": diversity,
                "scalar_rewards": scalar_rewards,
                "reward_vectors": evaluation_vectors,
                "rollouts": rollout_records,
            }

            predictions_file.write(
                json.dumps(prediction_record) + "\n"
            )

    rewards_array = np.asarray(
        all_reward_vectors,
        dtype=np.float64,
    )

    metrics = {
        "num_mazes": len(dataset),
        "routes_per_rollout": args.routes_per_rollout,
        "rollouts_per_maze": rollouts_per_maze,
        "candidates_per_maze": candidates_per_maze,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        **{
            f"best@{k}": float(np.mean(values))
            for k, values in per_k_values.items()
        },
        f"diversity@{max_k}": float(
            np.mean(per_maze_diversities)
        ),
        "format_valid_rollout_rate": float(
            np.mean(all_format_validities)
        ),
        "route_completion_rate": float(
            rewards_array[:, 0].mean()
        ),
        "mean_reward_components": {
            name: float(rewards_array[:, index].mean())
            for index, name in enumerate(REWARD_NAMES)
        },
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print()
    print("Maze evaluation complete")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()