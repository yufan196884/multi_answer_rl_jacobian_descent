import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import yaml
from datasets import Dataset, DatasetDict

from dataset_processing import process_dataset
from maze_dataset import (
    DIRECTIONS,
    build_maze_dataset,
    generate_surviving_mazes,
    maze_from_record,
    simulate_route,
    validate_maze,
)
from reward_fns import (
    format_reward,
    uniqueness_reward,
    vpo_candidate_reward_vectors,
)
from system_prompts import get_sys_prompt_formatted


REPO_ROOT = Path(__file__).resolve().parents[1]
MAZE_RECORD_FIELDS = (
    "seed",
    "open_cells",
    "start",
    "end",
    "gold_corner",
    "diamond_corner",
    "gold_cells",
    "diamond_cells",
    "lava_cells",
    "bonus_cell",
    "step_budget",
    "n_cycles",
    "via_gold",
    "via_diamond",
    "via_both_gold_first",
    "via_both_diamond_first",
)


def load_first_maze_record():
    maze = generate_surviving_mazes(start_seed=42, count=1)[0]
    return maze.to_record(num_routes=3)


def route_through(maze, waypoints):
    position = maze.start
    route = []

    for waypoint in [*waypoints, maze.end]:
        queue = deque([(position, [])])
        seen = {position}

        while queue:
            cell, path = queue.popleft()
            if cell == waypoint:
                route.extend(path)
                position = waypoint
                break

            for move, (row_delta, col_delta) in DIRECTIONS.items():
                neighbor = cell[0] + row_delta, cell[1] + col_delta
                if neighbor in maze.open_cells and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, [*path, move]))
        else:
            raise AssertionError(f"No route from {position} to {waypoint}")

    return route


def completion_for(routes):
    content = "\n".join(
        f"<route_{index}>{' '.join(route)}</route_{index}>"
        for index, route in enumerate(routes, start=1)
    )
    return [{"role": "assistant", "content": content}]


class MazeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_first_maze_record()
        cls.maze = maze_from_record(cls.record)
        cls.routes = [
            route_through(cls.maze, []),
            route_through(cls.maze, [cls.maze.gold_corner]),
            route_through(cls.maze, [cls.maze.bonus_cell]),
        ]
        cls.completions = [completion_for(cls.routes)]

    def test_maze_system_prompt_is_registered(self):
        prompt = get_sys_prompt_formatted("maze_multi_answer", 3)
        self.assertIn("Output exactly 3 routes", prompt)

    def test_maze_config_matches_reported_core_recipe(self):
        config_path = (
            REPO_ROOT
            / "configs"
            / "Qwen3-4B"
            / "vpo_multi_maze.yaml"
        )
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        self.assertEqual(config["model_name_or_path"], "Qwen/Qwen3-4B")
        self.assertEqual(config["num_candidates"], 3)
        self.assertEqual(config["num_generations"], 8)
        self.assertEqual(config["vpo_num_objectives"], 4)
        self.assertEqual(config["learning_rate"], 1e-6)
        self.assertEqual(config["beta"], 1e-3)
        self.assertEqual(config["epsilon"], 0.2)
        self.assertEqual(config["delta"], 3.0)

        rollout_batch = (
            4
            * config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
        )
        self.assertEqual(rollout_batch // config["num_generations"], 128)

    def test_generated_maze_splits_are_valid_and_disjoint(self):
        dataset = build_maze_dataset(train_size=10, test_size=3)
        splits = {}

        for split_name, expected_size in (("train", 10), ("test", 3)):
            rows = dataset[split_name]
            self.assertEqual(len(rows), expected_size)
            for row in rows:
                validate_maze(maze_from_record(row))
            splits[split_name] = {row["seed"] for row in rows}

        self.assertTrue(splits["train"].isdisjoint(splits["test"]))

    def test_dataset_processing_preserves_maze_metadata(self):
        dataset = DatasetDict({"train": Dataset.from_list([self.record])})
        args = SimpleNamespace(
            num_candidates=3,
            sys_prompt_name="maze_multi_answer",
            task_spec="gen",
            dataset_name="json",
            format_pattern="maze_multi_answer",
        )

        processed = process_dataset(dataset, args)["train"][0]

        self.assertEqual(processed["seed"], self.record["seed"])
        self.assertEqual(processed["open_cells"], self.record["open_cells"])
        self.assertEqual(processed["prompt"][0]["role"], "system")
        self.assertIn(self.record["question"], processed["prompt"][1]["content"])

    def test_maze_format_and_uniqueness_rewards(self):
        self.assertEqual(
            format_reward("maze_multi_answer", self.completions, 3),
            [1.0],
        )
        self.assertEqual(
            uniqueness_reward("maze_multi_answer", self.completions, 3),
            [1.0],
        )

        duplicate = [completion_for([self.routes[0]] * 3)]
        self.assertEqual(
            uniqueness_reward("maze_multi_answer", duplicate, 3),
            [0.0],
        )

    def test_maze_vpo_vectors_match_simulator(self):
        metadata = {
            field: [self.record[field]]
            for field in MAZE_RECORD_FIELDS
        }
        vectors = vpo_candidate_reward_vectors(
            format_pattern="maze_multi_answer",
            completions=self.completions,
            num_candidates=3,
            vpo_num_objectives=4,
            vpo_objective_mode="maze",
            vpo_apply_format_gate=True,
            vpo_apply_uniqueness_gate=False,
            **metadata,
        )

        expected = [
            list(simulate_route(self.maze, route))
            for route in self.routes
        ]
        self.assertEqual(vectors, [expected])

    def test_invalid_maze_completion_gets_zero_vectors(self):
        metadata = {
            field: [self.record[field]]
            for field in MAZE_RECORD_FIELDS
        }
        vectors = vpo_candidate_reward_vectors(
            format_pattern="maze_multi_answer",
            completions=[[{"role": "assistant", "content": "not routes"}]],
            num_candidates=3,
            vpo_num_objectives=4,
            vpo_objective_mode="maze",
            vpo_apply_format_gate=True,
            vpo_apply_uniqueness_gate=False,
            **metadata,
        )

        self.assertEqual(vectors, [[[0.0] * 4 for _ in range(3)]])


if __name__ == "__main__":
    unittest.main()
