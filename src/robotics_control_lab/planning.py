"""Sampling-based and graph motion-planning primitives."""

from dataclasses import dataclass
import heapq
import math

import numpy as np


@dataclass(frozen=True)
class CircleObstacle:
    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(-1)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("circle center must contain two finite values")
        if self.radius < 0 or not np.isfinite(self.radius):
            raise ValueError("circle radius must be finite and non-negative")
        object.__setattr__(self, "center", center.copy())

    def contains(self, q: np.ndarray, margin: float = 0.0) -> bool:
        q = np.asarray(q, dtype=float).reshape(2)
        if margin < 0 or not np.isfinite(margin):
            raise ValueError("margin must be finite and non-negative")
        return float(np.linalg.norm(q - self.center)) <= self.radius + margin


def _validate_bounds(bounds):
    if len(bounds) != 2 or any(len(b) != 2 for b in bounds):
        raise ValueError("bounds must contain two (min, max) pairs")
    bounds = tuple((float(b[0]), float(b[1])) for b in bounds)
    if any(not np.isfinite(value) for b in bounds for value in b) or any(lo >= hi for lo, hi in bounds):
        raise ValueError("bounds must be finite and satisfy min < max")
    return bounds


def edge_collision_free(a: np.ndarray, b: np.ndarray, obstacles: list[CircleObstacle], step: float = 0.01) -> bool:
    a = np.asarray(a, dtype=float).reshape(2)
    b = np.asarray(b, dtype=float).reshape(2)
    if step <= 0 or not np.isfinite(step):
        raise ValueError("step must be positive and finite")
    distance = float(np.linalg.norm(b - a))
    samples = max(1, int(math.ceil(distance / step)))
    return all(
        not any(ob.contains(a + t * (b - a)) for ob in obstacles)
        for t in np.linspace(0.0, 1.0, samples + 1)
    )


def _validate_problem(start, goal, bounds):
    bounds = _validate_bounds(bounds)
    start = np.asarray(start, dtype=float).reshape(2)
    goal = np.asarray(goal, dtype=float).reshape(2)
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError("start and goal must be finite")
    if any(not (lo <= value <= hi) for value, (lo, hi) in zip(start, bounds)):
        raise ValueError("start is outside bounds")
    if any(not (lo <= value <= hi) for value, (lo, hi) in zip(goal, bounds)):
        raise ValueError("goal is outside bounds")
    return start, goal, bounds


def _reconstruct(nodes: list[np.ndarray], parents: list[int], idx: int) -> np.ndarray:
    path = []
    while idx >= 0:
        path.append(nodes[idx])
        idx = parents[idx]
    return np.asarray(path[::-1])


def rrt(start, goal, bounds, obstacles, *, iterations=2000, step_size=0.08, goal_bias=0.1, seed=0):
    """Plan a collision-free 2D path with deterministic RRT sampling."""
    if iterations <= 0 or step_size <= 0 or not 0 <= goal_bias <= 1:
        raise ValueError("invalid RRT parameters")
    start, goal, bounds = _validate_problem(start, goal, bounds)
    if any(ob.contains(start) or ob.contains(goal) for ob in obstacles):
        return None
    rng = np.random.default_rng(seed)
    nodes = [start]
    parents = [-1]
    for _ in range(iterations):
        sample = goal if rng.random() < goal_bias else np.array([rng.uniform(*bounds[0]), rng.uniform(*bounds[1])])
        nearest = int(np.argmin([np.linalg.norm(n - sample) for n in nodes]))
        direction = sample - nodes[nearest]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        new = nodes[nearest] + step_size * direction / norm if norm > step_size else sample
        if not all(bounds[i][0] <= new[i] <= bounds[i][1] for i in range(2)):
            continue
        if not edge_collision_free(nodes[nearest], new, obstacles):
            continue
        nodes.append(new)
        parents.append(nearest)
        if np.linalg.norm(new - goal) <= step_size and edge_collision_free(new, goal, obstacles):
            nodes.append(goal)
            parents.append(len(nodes) - 2)
            return _reconstruct(nodes, parents, len(nodes) - 1)
    return None


def rrt_star(start, goal, bounds, obstacles, *, iterations=3000, step_size=0.08, neighbor_radius=0.2, goal_bias=0.05, seed=0):
    """Deterministic 2D RRT* with local rewiring and collision checking."""
    if iterations <= 0 or step_size <= 0 or neighbor_radius <= 0 or not 0 <= goal_bias <= 1:
        raise ValueError("invalid RRT* parameters")
    start, goal, bounds = _validate_problem(start, goal, bounds)
    if any(ob.contains(start) or ob.contains(goal) for ob in obstacles):
        return None
    rng = np.random.default_rng(seed)
    nodes = [start]
    parents = [-1]
    costs = [0.0]
    best_goal: tuple[float, int] | None = None
    for _ in range(iterations):
        sample = goal if rng.random() < goal_bias else np.array([rng.uniform(*bounds[0]), rng.uniform(*bounds[1])])
        nearest = int(np.argmin([np.linalg.norm(n - sample) for n in nodes]))
        direction = sample - nodes[nearest]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        new = nodes[nearest] + step_size * direction / norm if norm > step_size else sample
        if not all(bounds[i][0] <= new[i] <= bounds[i][1] for i in range(2)):
            continue
        if not edge_collision_free(nodes[nearest], new, obstacles):
            continue
        near = [
            i for i, node in enumerate(nodes)
            if np.linalg.norm(node - new) <= neighbor_radius and edge_collision_free(node, new, obstacles)
        ]
        parent = min(near, key=lambda i: costs[i] + np.linalg.norm(nodes[i] - new), default=nearest)
        new_cost = costs[parent] + float(np.linalg.norm(nodes[parent] - new))
        nodes.append(new)
        parents.append(parent)
        costs.append(new_cost)
        new_idx = len(nodes) - 1
        for i in near:
            candidate = new_cost + float(np.linalg.norm(nodes[i] - new))
            if candidate + 1e-12 < costs[i]:
                parents[i] = new_idx
                costs[i] = candidate
        if np.linalg.norm(new - goal) <= step_size and edge_collision_free(new, goal, obstacles):
            goal_cost = new_cost + float(np.linalg.norm(goal - new))
            if best_goal is None:
                nodes.append(goal)
                parents.append(new_idx)
                costs.append(goal_cost)
                best_goal = (goal_cost, len(nodes) - 1)
            elif goal_cost < best_goal[0]:
                goal_idx = best_goal[1]
                parents[goal_idx] = new_idx
                costs[goal_idx] = goal_cost
                best_goal = (goal_cost, goal_idx)
    return None if best_goal is None else _reconstruct(nodes, parents, best_goal[1])


def astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """A* on a binary occupancy grid (0 free, nonzero occupied)."""
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty 2D array")
    rows, cols = grid.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError("start is outside grid")
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        raise ValueError("goal is outside grid")
    if grid[start] != 0 or grid[goal] != 0:
        return None
    frontier = [(0.0, start)]
    parent = {start: None}
    cost = {start: 0.0}
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        for dr, dc in neighbors:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or grid[nxt] != 0:
                continue
            new_cost = cost[current] + math.hypot(dr, dc)
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                h = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(frontier, (new_cost + h, nxt))
                parent[nxt] = current
    return None
