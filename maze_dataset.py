from __future__ import annotations

import argparse
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from datasets import Dataset, DatasetDict


GRID_SIZE = 9
CENTER = (4, 4)

CORNERS = (
    (0, 0),
    (0, 8),
    (8, 0),
    (8, 8),
)

DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

Coord = tuple[int, int]
RewardVector = tuple[float, float, float, float]


@dataclass(frozen=True)
class MazeSpec:
    """One generated maze and all metadata needed to score model routes."""

    seed: int
    open_cells: frozenset[Coord]

    start: Coord
    end: Coord
    gold_corner: Coord
    diamond_corner: Coord

    gold_cells: frozenset[Coord]
    diamond_cells: frozenset[Coord]
    lava_cells: frozenset[Coord]
    bonus_cell: Coord

    step_budget: int
    n_cycles: int

    via_gold: int
    via_diamond: int
    via_both_gold_first: int
    via_both_diamond_first: int

    def render_grid(self) -> list[str]:
        """Render the maze using the symbols shown in the paper."""
        rows: list[str] = []

        for row in range(GRID_SIZE):
            symbols: list[str] = []

            for col in range(GRID_SIZE):
                cell = (row, col)

                if cell == self.start:
                    symbol = "S"
                elif cell == self.end:
                    symbol = "E"
                elif cell == self.bonus_cell:
                    symbol = "B"
                elif cell in self.gold_cells:
                    symbol = "G"
                elif cell in self.diamond_cells:
                    symbol = "D"
                elif cell in self.lava_cells:
                    symbol = "L"
                elif cell in self.open_cells:
                    symbol = "."
                else:
                    symbol = "#"

                symbols.append(symbol)

            rows.append(" ".join(symbols))

        return rows

    def to_record(self, num_routes: int = 3) -> dict:
        """
        Convert the maze into a Hugging Face Dataset-compatible record.

        Coordinates are converted to lists because Apache Arrow handles nested
        lists more consistently than sets or tuples.
        """
        return {
            "id": f"maze-{self.seed}",
            "seed": self.seed,
            "question": render_maze_prompt(self, num_routes=num_routes),
            "grid": self.render_grid(),
            "open_cells": _coordinates_to_lists(self.open_cells),
            "start": list(self.start),
            "end": list(self.end),
            "gold_corner": list(self.gold_corner),
            "diamond_corner": list(self.diamond_corner),
            "gold_cells": _coordinates_to_lists(self.gold_cells),
            "diamond_cells": _coordinates_to_lists(self.diamond_cells),
            "lava_cells": _coordinates_to_lists(self.lava_cells),
            "bonus_cell": list(self.bonus_cell),
            "step_budget": self.step_budget,
            "n_gold": len(self.gold_cells),
            "n_diamond": len(self.diamond_cells),
            "n_lava": len(self.lava_cells),
            "n_cycles": self.n_cycles,
            "via_gold": self.via_gold,
            "via_diamond": self.via_diamond,
            "via_both_gold_first": self.via_both_gold_first,
            "via_both_diamond_first": self.via_both_diamond_first,
        }


def _coordinates_to_lists(
    coordinates: Iterable[Coord],
) -> list[list[int]]:
    return [list(cell) for cell in sorted(coordinates)]


def maze_from_record(record: dict) -> MazeSpec:
    """
    Reconstruct MazeSpec from a Dataset row.

    This is useful inside reward functions, where the trainer passes dataset
    metadata as ordinary Python lists.
    """

    def coord(value: Sequence[int]) -> Coord:
        if len(value) != 2:
            raise ValueError(f"Invalid coordinate: {value}")
        return int(value[0]), int(value[1])

    def coord_set(values: Sequence[Sequence[int]]) -> frozenset[Coord]:
        return frozenset(coord(value) for value in values)

    return MazeSpec(
        seed=int(record["seed"]),
        open_cells=coord_set(record["open_cells"]),
        start=coord(record["start"]),
        end=coord(record["end"]),
        gold_corner=coord(record["gold_corner"]),
        diamond_corner=coord(record["diamond_corner"]),
        gold_cells=coord_set(record["gold_cells"]),
        diamond_cells=coord_set(record["diamond_cells"]),
        lava_cells=coord_set(record["lava_cells"]),
        bonus_cell=coord(record["bonus_cell"]),
        step_budget=int(record["step_budget"]),
        n_cycles=int(record["n_cycles"]),
        via_gold=int(record["via_gold"]),
        via_diamond=int(record["via_diamond"]),
        via_both_gold_first=int(record["via_both_gold_first"]),
        via_both_diamond_first=int(
            record["via_both_diamond_first"]
        ),
    )


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _all_cells() -> Iterator[Coord]:
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            yield row, col


def _neighbors(cell: Coord) -> Iterator[Coord]:
    row, col = cell

    for row_delta, col_delta in DIRECTIONS.values():
        next_row = row + row_delta
        next_col = col + col_delta

        if (
            0 <= next_row < GRID_SIZE
            and 0 <= next_col < GRID_SIZE
        ):
            yield next_row, next_col


def _bfs_distance(
    open_cells: set[Coord] | frozenset[Coord],
    start: Coord,
    end: Coord,
    blocked: set[Coord] | frozenset[Coord] = frozenset(),
) -> int | None:
    """Return the shortest path length, or None if no path exists."""
    if (
        start not in open_cells
        or end not in open_cells
        or start in blocked
        or end in blocked
    ):
        return None

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        cell, distance = queue.popleft()

        if cell == end:
            return distance

        for neighbor in _neighbors(cell):
            if (
                neighbor in open_cells
                and neighbor not in blocked
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return None


def _required_distance(
    open_cells: set[Coord],
    start: Coord,
    end: Coord,
) -> int:
    distance = _bfs_distance(open_cells, start, end)

    if distance is None:
        raise ValueError(f"No path from {start} to {end}.")

    return distance


def _manhattan_ball(
    center: Coord,
    radius: int,
) -> list[Coord]:
    center_row, center_col = center

    return [
        cell
        for cell in _all_cells()
        if (
            abs(cell[0] - center_row)
            + abs(cell[1] - center_col)
            <= radius
        )
    ]


# ---------------------------------------------------------------------------
# Maze generation
# ---------------------------------------------------------------------------

def _carve_prim_tree(
    rng: random.Random,
) -> set[Coord]:
    """
    Carve the initial acyclic maze using the rule from Appendix A.1.

    Start with an all-wall grid. Select one random seed cell. Repeatedly select
    a random frontier wall, and open it only if it has exactly one open
    four-neighbor.
    """
    seed_cell = (
        rng.randrange(GRID_SIZE),
        rng.randrange(GRID_SIZE),
    )

    open_cells = {seed_cell}

    frontier = list(_neighbors(seed_cell))
    frontier_set = set(frontier)

    while frontier:
        index = rng.randrange(len(frontier))
        cell = frontier.pop(index)
        frontier_set.remove(cell)

        open_neighbor_count = sum(
            neighbor in open_cells
            for neighbor in _neighbors(cell)
        )

        if open_neighbor_count != 1:
            continue

        open_cells.add(cell)

        for neighbor in _neighbors(cell):
            if (
                neighbor not in open_cells
                and neighbor not in frontier_set
            ):
                frontier.append(neighbor)
                frontier_set.add(neighbor)

    return open_cells


def _inject_cycles(
    open_cells: set[Coord],
    rng: random.Random,
    n_cycles: int,
) -> bool:
    """
    Add extra openings to create multiple routes.

    Each selected wall must currently have at least two open neighbors.
    """
    for _ in range(n_cycles):
        eligible_walls = [
            cell
            for cell in _all_cells()
            if (
                cell not in open_cells
                and sum(
                    neighbor in open_cells
                    for neighbor in _neighbors(cell)
                )
                >= 2
            )
        ]

        if not eligible_walls:
            return False

        open_cells.add(rng.choice(eligible_walls))

    return True


def _sample_open_cells(
    *,
    open_cells: set[Coord],
    candidates: Sequence[Coord],
    count: int,
    forbidden: set[Coord],
    rng: random.Random,
) -> set[Coord] | None:
    available = [
        cell
        for cell in candidates
        if cell in open_cells and cell not in forbidden
    ]

    if len(available) < count:
        return None

    return set(rng.sample(available, count))


def generate_maze_example(
    seed: int,
    *,
    check_both_item_orders: bool = True,
) -> MazeSpec | None:
    """
    Generate one candidate maze.

    Returns None if the maze fails one of the rejection conditions. The dataset
    builder scans consecutive seeds and keeps only surviving mazes.
    """
    rng = random.Random(seed)

    # Stage 1: Prim-style spanning tree.
    open_cells = _carve_prim_tree(rng)

    # Stage 2: add 18--28 cycle-generating openings.
    n_cycles = rng.randint(18, 28)

    if not _inject_cycles(open_cells, rng, n_cycles):
        return None

    # The four corners and center must be walkable for the stated task.
    # Reject rather than silently carving extra unreported cells.
    if not all(corner in open_cells for corner in CORNERS):
        return None

    if CENTER not in open_cells:
        return None

    # Choose one of the two diagonal start/end configurations.
    if rng.randrange(2) == 0:
        start = (0, 0)
        end = (8, 8)
        item_corners = [(0, 8), (8, 0)]
    else:
        start = (0, 8)
        end = (8, 0)
        item_corners = [(0, 0), (8, 8)]

    # Randomly assign the remaining corners to gold and diamond.
    rng.shuffle(item_corners)
    gold_corner, diamond_corner = item_corners

    start_to_gold = _required_distance(
        open_cells,
        start,
        gold_corner,
    )
    start_to_diamond = _required_distance(
        open_cells,
        start,
        diamond_corner,
    )
    gold_to_diamond = _required_distance(
        open_cells,
        gold_corner,
        diamond_corner,
    )
    gold_to_end = _required_distance(
        open_cells,
        gold_corner,
        end,
    )
    diamond_to_end = _required_distance(
        open_cells,
        diamond_corner,
        end,
    )

    via_gold = start_to_gold + gold_to_end
    via_diamond = start_to_diamond + diamond_to_end

    via_both_gold_first = (
        start_to_gold
        + gold_to_diamond
        + diamond_to_end
    )

    via_both_diamond_first = (
        start_to_diamond
        + gold_to_diamond
        + gold_to_end
    )

    step_budget = max(via_gold, via_diamond) + 7

    # Literal rejection condition printed in the appendix.
    if via_both_gold_first <= step_budget:
        return None

    # The prose says that no ordering visiting both corners should fit.
    if (
        check_both_item_orders
        and via_both_diamond_first <= step_budget
    ):
        return None

    n_gold = rng.randint(3, 5)
    n_diamond = rng.randint(3, 5)
    n_lava = rng.randint(3, 5)

    forbidden = {
        start,
        end,
        CENTER,
    }

    gold_cells = _sample_open_cells(
        open_cells=open_cells,
        candidates=_manhattan_ball(
            gold_corner,
            radius=2,
        ),
        count=n_gold,
        forbidden=forbidden,
        rng=rng,
    )

    if gold_cells is None:
        return None

    forbidden.update(gold_cells)

    diamond_cells = _sample_open_cells(
        open_cells=open_cells,
        candidates=_manhattan_ball(
            diamond_corner,
            radius=2,
        ),
        count=n_diamond,
        forbidden=forbidden,
        rng=rng,
    )

    if diamond_cells is None:
        return None

    forbidden.update(diamond_cells)

    interior_cells = [
        (row, col)
        for row in range(2, 7)
        for col in range(2, 7)
    ]

    lava_cells = _sample_open_cells(
        open_cells=open_cells,
        candidates=interior_cells,
        count=n_lava,
        forbidden=forbidden,
        rng=rng,
    )

    if lava_cells is None:
        return None

    # Final rejection condition: a lava-free path to E must fit the budget.
    lava_avoiding_distance = _bfs_distance(
        open_cells,
        start,
        end,
        blocked=lava_cells,
    )

    if (
        lava_avoiding_distance is None
        or lava_avoiding_distance > step_budget
    ):
        return None

    return MazeSpec(
        seed=seed,
        open_cells=frozenset(open_cells),
        start=start,
        end=end,
        gold_corner=gold_corner,
        diamond_corner=diamond_corner,
        gold_cells=frozenset(gold_cells),
        diamond_cells=frozenset(diamond_cells),
        lava_cells=frozenset(lava_cells),
        bonus_cell=CENTER,
        step_budget=step_budget,
        n_cycles=n_cycles,
        via_gold=via_gold,
        via_diamond=via_diamond,
        via_both_gold_first=via_both_gold_first,
        via_both_diamond_first=via_both_diamond_first,
    )


def generate_surviving_mazes(
    *,
    start_seed: int,
    count: int,
    check_both_item_orders: bool = True,
    max_attempts: int = 1_000_000,
) -> list[MazeSpec]:
    """
    Return the first `count` surviving mazes from consecutive candidate seeds.

    This is important: the paper says the first 1000 mazes that survive,
    rather than saying that seeds 42 through 1041 all survive.
    """
    if count < 0:
        raise ValueError("count must be non-negative.")

    mazes: list[MazeSpec] = []
    candidate_seed = start_seed
    attempts = 0

    while len(mazes) < count:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Only generated {len(mazes)} surviving mazes "
                f"after {max_attempts} candidate seeds."
            )

        maze = generate_maze_example(
            candidate_seed,
            check_both_item_orders=check_both_item_orders,
        )

        if maze is not None:
            mazes.append(maze)

        candidate_seed += 1
        attempts += 1

    return mazes


def build_maze_dataset(
    train_size: int = 1000,
    test_size: int = 100,
    *,
    train_start_seed: int = 42,
    test_start_seed: int = 4242,
    num_routes: int = 3,
    check_both_item_orders: bool = True,
) -> DatasetDict:
    """Build the paper's training and held-out test splits."""
    train_mazes = generate_surviving_mazes(
        start_seed=train_start_seed,
        count=train_size,
        check_both_item_orders=check_both_item_orders,
    )

    test_mazes = generate_surviving_mazes(
        start_seed=test_start_seed,
        count=test_size,
        check_both_item_orders=check_both_item_orders,
    )

    return DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    maze.to_record(num_routes=num_routes)
                    for maze in train_mazes
                ]
            ),
            "test": Dataset.from_list(
                [
                    maze.to_record(num_routes=num_routes)
                    for maze in test_mazes
                ]
            ),
        }
    )


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_maze_prompt(
    maze: MazeSpec,
    *,
    num_routes: int = 3,
) -> str:
    grid_text = "\n".join(maze.render_grid())

    route_tags = ", ".join(
        f"<route_{index}>...</route_{index}>"
        for index in range(1, num_routes + 1)
    )

    return f"""Navigate a 9x9 maze from S to E. Collect gold and diamonds, avoid lava.

Grid:
{grid_text}

- Move: UP, DOWN, LEFT, RIGHT. # is a wall -- you cannot enter it.
- Do not leave the grid.
- Collect: G (Gold), D (Diamond), B (Bonus) tiles by stepping on them.
- Avoid: L (Lava) tiles. Stepping on lava costs you.
- Visiting a B cell multiplies your other scores -- explore!
- You MUST reach E. If you do not reach E, your score is zero everywhere.
- Items only count if collected BEFORE you reach E; the trajectory ends at E.
- You have {maze.step_budget} steps per route.

This maze has {len(maze.gold_cells)} Gold, \
{len(maze.diamond_cells)} Diamond, \
{len(maze.lava_cells)} Lava, and 1 Bonus tile.

Reason briefly about the maze, then provide {num_routes} genuinely different
routes from S to E. Each route must be a space-separated sequence of
UP/DOWN/LEFT/RIGHT moves.

Inside each route tag, put ONLY moves. Put any reasoning outside the tags.
Required route tags: {route_tags}
"""


# ---------------------------------------------------------------------------
# Output parsing and reward simulation
# ---------------------------------------------------------------------------

def parse_moves(route_text: str) -> list[str] | None:
    """
    Parse a route strictly.

    Unknown tokens invalidate the route rather than being silently discarded.
    """
    tokens = route_text.strip().split()

    if not tokens:
        return None

    moves = [token.upper() for token in tokens]

    if any(move not in DIRECTIONS for move in moves):
        return None

    return moves


def extract_numbered_routes(
    completion: str,
    *,
    num_routes: int = 3,
) -> list[list[str]] | None:
    """Extract <route_1> through <route_K> from one model completion."""
    routes: list[list[str]] = []

    for index in range(1, num_routes + 1):
        match = re.search(
            rf"<route_{index}>(.*?)</route_{index}>",
            completion,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match is None:
            return None

        moves = parse_moves(match.group(1))

        if moves is None:
            return None

        routes.append(moves)

    return routes


def simulate_route(
    maze: MazeSpec,
    moves: Sequence[str],
) -> RewardVector:
    """
    Return:

        (completion, gold_rate, diamond_rate, lava_avoidance)

    Attempting to walk through a wall or outside the grid consumes one step but
    leaves the agent in its current position.
    """
    position = maze.start

    visited_gold: set[Coord] = set()
    visited_diamond: set[Coord] = set()
    visited_lava: set[Coord] = set()

    # Only the first step_budget actions are executed.
    for raw_move in moves[: maze.step_budget]:
        move = raw_move.upper()

        if move not in DIRECTIONS:
            return 0.0, 0.0, 0.0, 0.0

        row_delta, col_delta = DIRECTIONS[move]

        next_position = (
            position[0] + row_delta,
            position[1] + col_delta,
        )

        # Borders and walls block movement.
        if next_position in maze.open_cells:
            position = next_position

        # The trajectory ends immediately upon entering E.
        if position == maze.end:
            gold_rate = (
                len(visited_gold)
                / len(maze.gold_cells)
            )

            diamond_rate = (
                len(visited_diamond)
                / len(maze.diamond_cells)
            )

            lava_avoidance = 1.0 - (
                len(visited_lava)
                / len(maze.lava_cells)
            )

            return (
                1.0,
                max(0.0, min(1.0, gold_rate)),
                max(0.0, min(1.0, diamond_rate)),
                max(0.0, min(1.0, lava_avoidance)),
            )

        if position in maze.gold_cells:
            visited_gold.add(position)

        if position in maze.diamond_cells:
            visited_diamond.add(position)

        if position in maze.lava_cells:
            visited_lava.add(position)

        # The bonus tile intentionally has no effect.

    # Failure to reach E zeros every reward dimension.
    return 0.0, 0.0, 0.0, 0.0


def uniform_scalar_reward(
    reward_vector: Sequence[float],
) -> float:
    """The GRPO/evaluation scalar: mean of the four reward dimensions."""
    if len(reward_vector) != 4:
        raise ValueError(
            "Maze reward vectors must have exactly four components."
        )

    return sum(float(value) for value in reward_vector) / 4.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_maze(maze: MazeSpec) -> None:
    """Raise an exception if an accepted maze violates a core invariant."""
    if maze.start == maze.end:
        raise ValueError("Start and end must differ.")

    if not all(corner in maze.open_cells for corner in CORNERS):
        raise ValueError("All four corners must be open.")

    if maze.bonus_cell not in maze.open_cells:
        raise ValueError("The bonus cell must be open.")

    if maze.via_gold > maze.step_budget:
        raise ValueError("Gold detour does not fit within the budget.")

    if maze.via_diamond > maze.step_budget:
        raise ValueError("Diamond detour does not fit within the budget.")

    if maze.via_both_gold_first <= maze.step_budget:
        raise ValueError(
            "Gold-then-diamond route incorrectly fits within the budget."
        )

    if maze.via_both_diamond_first <= maze.step_budget:
        raise ValueError(
            "Diamond-then-gold route incorrectly fits within the budget."
        )

    lava_free_distance = _bfs_distance(
        maze.open_cells,
        maze.start,
        maze.end,
        blocked=maze.lava_cells,
    )

    if (
        lava_free_distance is None
        or lava_free_distance > maze.step_budget
    ):
        raise ValueError(
            "No lava-free start-to-end route fits within the budget."
        )


# ---------------------------------------------------------------------------
# Command-line generation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the VPO 9x9 Maze dataset."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/maze"),
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num-routes",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--literal-paper-order",
        action="store_true",
        help=(
            "Only check the printed gold-then-diamond detour rather "
            "than checking both item visit orders."
        ),
    )

    args = parser.parse_args()

    dataset = build_maze_dataset(
        train_size=args.train_size,
        test_size=args.test_size,
        num_routes=args.num_routes,
        check_both_item_orders=not args.literal_paper_order,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = args.output_dir / "train.jsonl"
    test_path = args.output_dir / "test.jsonl"

    dataset["train"].to_json(train_path)
    dataset["test"].to_json(test_path)

    print(dataset)
    print(f"Train split written to: {train_path}")
    print(f"Test split written to:  {test_path}")
    print()
    print("First training example:")
    print(dataset["train"][0]["question"])


if __name__ == "__main__":
    main()