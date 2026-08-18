"""Sampling-based and graph motion-planning primitives."""

from dataclasses import dataclass
import heapq
import math
import numpy as np


@dataclass(frozen=True)
class CircleObstacle:
    center: np.ndarray
    radius: float

    def contains(self, q: np.ndarray, margin: float = 0.0) -> bool:
        return float(np.linalg.norm(np.asarray(q) - self.center)) <= self.radius + margin


def edge_collision_free(a: np.ndarray, b: np.ndarray, obstacles: list[CircleObstacle], step: float = 0.01) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if step <= 0:
        raise ValueError("step must be positive")
    distance = float(np.linalg.norm(b - a))
    samples = max(1, int(math.ceil(distance / step)))
    for t in np.linspace(0.0, 1.0, samples + 1):
        q = a + t * (b - a)
        if any(ob.contains(q) for ob in obstacles):
            return False
    return True


def rrt(
    start: np.ndarray,
    goal: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    obstacles: list[CircleObstacle],
    *,
    iterations: int = 2000,
    step_size: float = 0.08,
    goal_bias: float = 0.1,
    seed: int = 0,
) -> np.ndarray | None:
    """Plan a collision-free 2D path with deterministic RRT sampling."""
    if iterations <= 0 or step_size <= 0 or not 0 <= goal_bias <= 1:
        raise ValueError("invalid RRT parameters")
    rng = np.random.default_rng(seed)
    start = np.asarray(start, dtype=float).reshape(2)
    goal = np.asarray(goal, dtype=float).reshape(2)
    nodes = [start]
    parents = [-1]
    for _ in range(iterations):
        sample = goal if rng.random() < goal_bias else np.array([rng.uniform(*bounds[0]), rng.uniform(*bounds[1])])
        distances = [np.linalg.norm(n - sample) for n in nodes]
        nearest = int(np.argmin(distances))
        direction = sample - nodes[nearest]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        new = nodes[nearest] + step_size * direction / norm if norm > step_size else sample
        if not all(bounds[i][0] <= new[i] <= bounds[i][1] for i in range(2)):
            continue
        if not edge_collision_free(nodes[nearest], new, obstacles):
            continue
        nodes.append(new); parents.append(nearest)
        if np.linalg.norm(new - goal) <= step_size and edge_collision_free(new, goal, obstacles):
            nodes.append(goal); parents.append(len(nodes) - 2)
            path = []
            idx = len(nodes) - 1
            while idx >= 0:
                path.append(nodes[idx]); idx = parents[idx]
            return np.asarray(path[::-1])
    return None


def astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """A* on a binary occupancy grid (0 free, nonzero occupied)."""
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")
    if grid[start] != 0 or grid[goal] != 0:
        return None
    frontier = [(0.0, start)]
    parent = {start: None}
    cost = {start: 0.0}
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            while current is not None:
                path.append(current); current = parent[current]
            return path[::-1]
        for dr, dc in neighbors:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]) or grid[nxt] != 0:
                continue
            new_cost = cost[current] + math.hypot(dr, dc)
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                h = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(frontier, (new_cost + h, nxt)); parent[nxt] = current
    return None
