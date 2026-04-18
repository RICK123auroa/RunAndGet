from __future__ import annotations 

import json
import random
import re
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import config

try:
    import pygame
except ImportError:
    print("这个游戏需要下载pygame，作者之后会写一个bat脚本")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("这个游戏需要下载numpy，作者之后会写一个bat脚本")
    sys.exit(1)

Vec2 = tuple[int, int]
DIRECTIONS_8: tuple[Vec2, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

DIRECTION_LABELS: dict[Vec2, str] = {
    (-1, 0): "L",
    (1, 0): "R",
    (0, -1): "U",
    (0, 1): "D",
    (-1, -1): "UL",
    (-1, 1): "DL",
    (1, -1): "UR",
    (1, 1): "DR",
    (0, 0): "HERE",
}


@dataclass
class Monster:
    pos: Vec2
    spawned_at: float
    move_progress: float = 0.0
    active: bool = True


@dataclass
class BuffNode:
    pos: Vec2
    active: bool = True
    cooldown_left: float = 0.0


def parse_text_rows(text: str) -> list[list[int]] | None:
    rows: list[list[int]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if set(line).issubset({"0", "1"}):
            row = [int(ch) for ch in line]
        else:
            tokens = [token for token in re.split(r"[^01]+", line) if token]
            if not tokens:
                continue
            row = [int(token) for token in tokens]
        rows.append(row)

    if not rows:
        return None

    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return None

    return rows


def matrix_from_json_value(value: object) -> list[list[int]] | None:
    if not isinstance(value, list) or not value:
        return None

    matrix: list[list[int]] = []
    for row_value in value:
        if not isinstance(row_value, list) or not row_value:
            return None
        row: list[int] = []
        for cell in row_value:
            if cell in (0, 1):
                row.append(int(cell))
            elif isinstance(cell, bool):
                row.append(1 if cell else 0)
            else:
                return None
        matrix.append(row)

    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        return None

    return matrix


def try_load_matrix_from_file(path: Path) -> list[list[int]] | None:
    suffix = path.suffix.lower()

    if suffix == ".npy" and np is not None:
        try:
            array = np.load(path, allow_pickle=False)
        except Exception:
            return None
        if array.ndim < 2:
            return None
        while array.ndim > 2:
            array = array[0]
        matrix: list[list[int]] = []
        for row in array.tolist():
            if not isinstance(row, list):
                return None
            out_row: list[int] = []
            for cell in row:
                try:
                    value = int(cell)
                except Exception:
                    return None
                if value not in (0, 1):
                    return None
                out_row.append(value)
            matrix.append(out_row)
        if not matrix:
            return None
        width = len(matrix[0])
        if width == 0 or any(len(row) != width for row in matrix):
            return None
        return matrix

    if suffix == ".json":
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return matrix_from_json_value(parsed)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="gbk")
        except Exception:
            return None
    except Exception:
        return None

    return parse_text_rows(text)


def generate_fallback_grid(size: int, density: float) -> list[list[int]]:
    grid: list[list[int]] = []
    for y in range(size):
        row: list[int] = []
        for x in range(size):
            if x == 0 or y == 0 or x == size - 1 or y == size - 1:
                row.append(1)
            else:
                row.append(1 if random.random() < density else 0)
        grid.append(row)

    cx = size // 2
    cy = size // 2
    x, y = cx, cy
    grid[y][x] = 0
    for _ in range(size * size * 3):
        dx, dy = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        x = max(1, min(size - 2, x + dx))
        y = max(1, min(size - 2, y + dy))
        grid[y][x] = 0

    return grid


def save_fallback_grid_file(tensor_dir: Path, grid: list[list[int]]) -> None:
    tensor_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = tensor_dir / "fallback_map.txt"
    text = "\n".join(" ".join(str(cell) for cell in row) for row in grid)
    fallback_path.write_text(text, encoding="utf-8")


def load_or_generate_grid(tensor_dir: Path, map_size: int) -> tuple[list[list[int]], str, int | None]:
    tensor_dir.mkdir(parents=True, exist_ok=True)

    files = [
        path
        for path in sorted(tensor_dir.iterdir(), key=lambda p: p.name)
        if path.is_file() and not path.name.startswith(".")
    ]
    random_files = files[:]
    random.shuffle(random_files)

    stacked_rows: list[list[int]] = []
    for path in random_files:
        matrix = try_load_matrix_from_file(path)
        if not matrix:
            continue

        height = len(matrix)
        width = len(matrix[0])

        if height >= map_size and width >= map_size:
            cropped = [row[:map_size] for row in matrix[:map_size]]
            map_number = files.index(path) + 1
            return cropped, path.name, map_number

        if width >= map_size and height < map_size:
            stacked_rows.extend([row[:map_size] for row in matrix])
            if len(stacked_rows) >= map_size:
                return stacked_rows[:map_size], "stacked-files", None

    if len(stacked_rows) >= map_size:
        return stacked_rows[:map_size], "stacked-files", None

    fallback = generate_fallback_grid(map_size, config.FALLBACK_OBSTACLE_DENSITY)
    save_fallback_grid_file(tensor_dir, fallback)
    return fallback, "fallback_map.txt", None


class TreasureHideGame:
    def __init__(self) -> None:
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.tensor_dir = self.project_root / "地图" / "tensorImg"
        self.replay_root = self.project_root / "录像文件"
        self.replay_root.mkdir(parents=True, exist_ok=True)

        self.grid, self.map_source, self.map_number = load_or_generate_grid(self.tensor_dir, config.MAP_SIZE)
        if self.map_number is not None:
            self.map_title = f"第{self.map_number}号地图"
        else:
            match = re.search(r"\d+", self.map_source)
            self.map_title = f"第{match.group()}号地图" if match else self.map_source
        self.passable_cells = [
            (x, y)
            for y in range(config.MAP_SIZE)
            for x in range(config.MAP_SIZE)
            if self.grid[y][x] == 0
        ]
        if not self.passable_cells:
            raise RuntimeError("No walkable cells in map")

        pygame.init()
        pygame.display.set_caption("Treasure Hide & Monster Chase")

        self.view_pixels = config.VIEW_SIZE * config.CELL_SIZE
        self.side_panel_width = config.SIDE_PANEL_WIDTH
        self.hud_height = 120
        self.screen = pygame.display.set_mode(
            (self.view_pixels + self.side_panel_width, self.view_pixels + self.hud_height)
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.small_font = pygame.font.SysFont("consolas", 17)
        self.big_font = pygame.font.SysFont("consolas", 34)

        self.player_pos: Vec2 = self._pick_random_passable(set())
        self.player_move_progress = 0.0
        self.last_move_dir: Vec2 = (0, 1)
        self.player_buff_left = 0.0
        self.last_dash_used_at = -config.DASH_COOLDOWN_SECONDS

        monster1_pos = self._pick_spawn_near_player(
            excluded={self.player_pos},
            min_distance=config.MONSTER_SPAWN_MIN_DISTANCE,
            max_distance=config.MONSTER_SPAWN_MAX_DISTANCE,
        )
        self.monster1 = Monster(pos=monster1_pos, spawned_at=0.0)
        self.monster2: Monster | None = None
        self.monsters_sped_up = False

        reserved = {self.player_pos, self.monster1.pos}
        self.chests = self._spawn_chests(reserved)
        reserved.update(self.chests)
        self.buff_nodes = self._spawn_buffs(reserved)

        self.elapsed_seconds = 0.0
        self.score = 0.0
        self.last_recorded_second = -1
        self.records: list[dict[str, object]] = []

        self.started = False
        self.finished = False
        self.win = False
        self.end_reason = ""
        self.replay_written = False
        self.replay_folder: Path | None = None

    def _spawn_chests(self, excluded: set[Vec2]) -> set[Vec2]:
        count = random.randint(config.CHEST_MIN_COUNT, config.CHEST_MAX_COUNT)
        available = [cell for cell in self.passable_cells if cell not in excluded]
        if not available:
            return set()
        count = min(count, len(available))
        return set(random.sample(available, count))

    def _spawn_buffs(self, excluded: set[Vec2]) -> list[BuffNode]:
        available = [cell for cell in self.passable_cells if cell not in excluded]
        if not available:
            return []
        count = min(config.BUFF_COUNT, len(available))
        buff_cells = random.sample(available, count)
        return [BuffNode(pos=cell) for cell in buff_cells]

    def _pick_random_passable(self, excluded: set[Vec2]) -> Vec2:
        choices = [cell for cell in self.passable_cells if cell not in excluded]
        if not choices:
            return random.choice(self.passable_cells)
        return random.choice(choices)

    def _pick_spawn_near_player(
        self,
        excluded: set[Vec2],
        min_distance: int,
        max_distance: int,
        center: Vec2 | None = None,
    ) -> Vec2:
        if center is None:
            center = self.player_pos
        cx, cy = center
        candidates: list[Vec2] = []
        for y in range(max(0, cy - max_distance), min(config.MAP_SIZE, cy + max_distance + 1)):
            for x in range(max(0, cx - max_distance), min(config.MAP_SIZE, cx + max_distance + 1)):
                distance = max(abs(x - cx), abs(y - cy))
                if distance < min_distance or distance > max_distance:
                    continue
                if not self.is_walkable((x, y)):
                    continue
                if (x, y) in excluded:
                    continue
                candidates.append((x, y))
        if candidates:
            return random.choice(candidates)
        return self._pick_random_passable(excluded)

    def is_walkable(self, pos: Vec2) -> bool:
        x, y = pos
        if x < 0 or y < 0 or x >= config.MAP_SIZE or y >= config.MAP_SIZE:
            return False
        return self.grid[y][x] == 0

    @staticmethod
    def neighbors(pos: Vec2) -> Iterable[Vec2]:
        x, y = pos
        for dx, dy in DIRECTIONS_8:
            yield (x + dx, y + dy)

    def input_direction(self) -> Vec2:
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]

        dx = int(right) - int(left)
        dy = int(down) - int(up)
        return (dx, dy)

    def try_step(self, start: Vec2, direction: Vec2) -> Vec2:
        if direction == (0, 0):
            return start

        sx, sy = start
        dx, dy = direction

        direct = (sx + dx, sy + dy)
        if self.is_walkable(direct):
            return direct

        if dx != 0 and dy != 0:
            horizontal = (sx + dx, sy)
            vertical = (sx, sy + dy)
            if self.is_walkable(horizontal):
                return horizontal
            if self.is_walkable(vertical):
                return vertical

        return start

    def current_player_speed(self) -> float:
        multiplier = config.BUFF_SPEED_MULTIPLIER if self.player_buff_left > 0 else 1.0
        return config.PLAYER_BASE_SPEED * multiplier

    def current_monster_speed(self) -> float:
        multiplier = config.MONSTER_SPEEDUP_MULTIPLIER if self.monsters_sped_up else 1.0
        return config.MONSTER_BASE_SPEED * multiplier

    def try_dash(self) -> None:
        if self.elapsed_seconds - self.last_dash_used_at < config.DASH_COOLDOWN_SECONDS:
            return

        direction = self.input_direction()
        if direction == (0, 0):
            direction = self.last_move_dir
        if direction == (0, 0):
            return

        px, py = self.player_pos
        dx, dy = direction
        target = self.player_pos

        # 闪现落点规则：
        # 从最远距离开始向回找，落在“最远且可落地”的格子。
        for step in range(config.DASH_DISTANCE, 0, -1):
            nx = px + dx * step
            ny = py + dy * step
            candidate = (nx, ny)
            if self.is_walkable(candidate):
                target = candidate
                break

        if target != self.player_pos:
            self.player_pos = target
            self.last_dash_used_at = self.elapsed_seconds
            self.collect_items_on_player()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.end_reason = "quit"
                self.finish_game(force_write=True)
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.end_reason = "quit"
                    self.finish_game(force_write=True)
                    return False
                if (not self.finished) and (not self.started) and event.key == pygame.K_SPACE:
                    self.started = True
                    continue
                if self.started and (not self.finished) and event.key == pygame.K_SPACE:
                    self.try_dash()
        return True

    def update_player(self, dt: float) -> None:
        direction = self.input_direction()
        if direction != (0, 0):
            self.last_move_dir = direction
            self.player_move_progress += self.current_player_speed() * dt

            while self.player_move_progress >= 1.0:
                next_pos = self.try_step(self.player_pos, direction)
                if next_pos == self.player_pos:
                    self.player_move_progress = 0.0
                    break
                self.player_move_progress -= 1.0
                self.player_pos = next_pos
                self.score += config.PLAYER_STEP_SCORE
                self.collect_items_on_player()
        else:
            self.player_move_progress = min(self.player_move_progress, 1.0)

    def bfs_shortest_path(self, start: Vec2, goal: Vec2) -> list[Vec2]:
        if start == goal:
            return [start]

        queue: deque[Vec2] = deque([start])
        parent: dict[Vec2, Vec2 | None] = {start: None}

        while queue:
            current = queue.popleft()
            for nxt in self.neighbors(current):
                if nxt in parent:
                    continue
                if not self.is_walkable(nxt):
                    continue
                parent[nxt] = current
                if nxt == goal:
                    queue.clear()
                    break
                queue.append(nxt)

        if goal not in parent:
            return []

        path: list[Vec2] = []
        cur: Vec2 | None = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def next_monster_step(self, pos: Vec2) -> Vec2:
        path = self.bfs_shortest_path(pos, self.player_pos)
        if len(path) >= 2:
            return path[1]
        return pos

    def active_monsters(self) -> list[Monster]:
        monsters = [self.monster1]
        if self.monster2 is not None and self.monster2.active:
            monsters.append(self.monster2)
        return monsters

    def update_monsters(self, dt: float) -> None:
        speed = self.current_monster_speed()

        for monster in self.active_monsters():
            monster.move_progress += speed * dt
            while monster.move_progress >= 1.0:
                monster.move_progress -= 1.0
                next_pos = self.next_monster_step(monster.pos)
                if next_pos == monster.pos:
                    monster.move_progress = 0.0
                    break
                monster.pos = next_pos

    def update_buffs(self, dt: float) -> None:
        for buff in self.buff_nodes:
            if not buff.active:
                buff.cooldown_left = max(0.0, buff.cooldown_left - dt)
                if buff.cooldown_left <= 0.0:
                    buff.active = True

    def collect_items_on_player(self) -> None:
        if self.player_pos in self.chests:
            self.chests.remove(self.player_pos)
            self.score += config.CHEST_SCORE

        for buff in self.buff_nodes:
            if buff.active and buff.pos == self.player_pos:
                buff.active = False
                buff.cooldown_left = config.BUFF_RESPAWN_SECONDS
                self.player_buff_left = max(self.player_buff_left, config.BUFF_DURATION_SECONDS)

    def maybe_spawn_second_monster(self) -> None:
        if self.monster2 is not None:
            return
        if self.elapsed_seconds < config.MONSTER2_SPAWN_TIME_SECONDS:
            return

        exclude = {self.player_pos, self.monster1.pos}
        spawn = self._pick_spawn_near_player(
            excluded=exclude,
            min_distance=config.MONSTER_SPAWN_MIN_DISTANCE,
            max_distance=config.MONSTER_SPAWN_MAX_DISTANCE,
        )
        self.monster2 = Monster(pos=spawn, spawned_at=self.elapsed_seconds)

    def update_score_and_recording(self) -> None:
        whole_seconds = int(self.elapsed_seconds)

        while self.last_recorded_second < whole_seconds:
            self.last_recorded_second += 1
            self.records.append(self.snapshot(self.last_recorded_second))

    def snapshot(self, second_mark: int) -> dict[str, object]:
        monster2_pos = self.monster2.pos if self.monster2 is not None else None
        return {
            "time": second_mark,
            "score": round(self.score, 3),
            "hero": list(self.player_pos),
            "monster1": list(self.monster1.pos),
            "monster2": list(monster2_pos) if monster2_pos is not None else None,
            "chests_left": len(self.chests),
            "chests": [list(cell) for cell in sorted(self.chests)],
            "buffs": [
                {
                    "pos": list(buff.pos),
                    "active": buff.active,
                    "cooldown_left": round(buff.cooldown_left, 3),
                }
                for buff in self.buff_nodes
            ],
        }

    def check_end_conditions(self) -> None:
        for monster in self.active_monsters():
            if monster.pos == self.player_pos:
                self.end_reason = "caught"
                self.finish_game(force_write=True)
                return

        if self.elapsed_seconds >= config.WIN_SURVIVAL_SECONDS:
            self.win = True
            self.end_reason = "survival"
            self.finish_game(force_write=True)

    def finish_game(self, force_write: bool = False) -> None:
        if self.finished and not force_write:
            return

        self.finished = True

        if not self.replay_written:
            end_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = self.replay_root / end_tag
            folder.mkdir(parents=True, exist_ok=True)

            map_path = folder / "map.json"
            record_path = folder / "record.jsonl"
            summary_path = folder / "summary.json"

            map_payload = {
                "map_size": config.MAP_SIZE,
                "grid": self.grid,
            }
            map_path.write_text(json.dumps(map_payload, ensure_ascii=False), encoding="utf-8")

            with record_path.open("w", encoding="utf-8") as f:
                for record in self.records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary = {
                "end_reason": self.end_reason,
                "win": self.win,
                "final_score": round(self.score, 3),
                "survival_seconds": round(self.elapsed_seconds, 3),
                "map_source": self.map_source,
                "map_title": self.map_title,
                "map_number": self.map_number,
                "map_file": map_path.name,
                "record_file": record_path.name,
                "hero_final": list(self.player_pos),
                "monster1_final": list(self.monster1.pos),
                "monster2_final": list(self.monster2.pos) if self.monster2 else None,
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            self.replay_folder = folder
            self.replay_written = True

    def update(self, dt: float) -> None:
        if self.finished:
            return
        if not self.started:
            return

        self.elapsed_seconds += dt

        if self.player_buff_left > 0:
            self.player_buff_left = max(0.0, self.player_buff_left - dt)

        if (not self.monsters_sped_up) and self.elapsed_seconds >= config.MONSTER_SPEEDUP_TIME_SECONDS:
            self.monsters_sped_up = True

        self.maybe_spawn_second_monster()
        self.update_buffs(dt)
        self.update_player(dt)
        self.update_monsters(dt)
        self.collect_items_on_player()
        self.update_score_and_recording()
        self.check_end_conditions()

    def world_to_screen(self, world_pos: Vec2) -> Vec2 | None:
        radius = config.VIEW_SIZE // 2
        origin_x = self.player_pos[0] - radius
        origin_y = self.player_pos[1] - radius

        wx, wy = world_pos
        sx = wx - origin_x
        sy = wy - origin_y

        if sx < 0 or sy < 0 or sx >= config.VIEW_SIZE or sy >= config.VIEW_SIZE:
            return None

        return sx * config.CELL_SIZE, sy * config.CELL_SIZE

    @staticmethod
    def _sign(value: int) -> int:
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    def direction_to(self, target: Vec2) -> Vec2:
        dx = self._sign(target[0] - self.player_pos[0])
        dy = self._sign(target[1] - self.player_pos[1])
        return (dx, dy)

    @staticmethod
    def draw_direction_arrow(
        surface: pygame.Surface,
        center: Vec2,
        direction: Vec2,
        color: tuple[int, int, int],
        size: int = 12,
    ) -> None:
        vector = pygame.Vector2(direction[0], direction[1])
        if vector.length_squared() == 0:
            pygame.draw.circle(surface, color, center, max(3, size // 3))
            return

        vector = vector.normalize()
        perp = pygame.Vector2(-vector.y, vector.x)
        center_vec = pygame.Vector2(center)

        tip = center_vec + vector * size
        base = center_vec - vector * (size * 0.58)
        left = base + perp * (size * 0.5)
        right = base - perp * (size * 0.5)
        tail = center_vec - vector * (size * 0.95)

        p_tail = (int(round(tail.x)), int(round(tail.y)))
        p_base = (int(round(base.x)), int(round(base.y)))
        p_tip = (int(round(tip.x)), int(round(tip.y)))
        p_left = (int(round(left.x)), int(round(left.y)))
        p_right = (int(round(right.x)), int(round(right.y)))

        pygame.draw.line(surface, color, p_tail, p_base, 3)
        pygame.draw.polygon(surface, color, [p_tip, p_left, p_right])

    def nearest_chest(self) -> Vec2 | None:
        if not self.chests:
            return None
        px, py = self.player_pos
        return min(self.chests, key=lambda c: max(abs(c[0] - px), abs(c[1] - py)))

    def draw_glow_dot(self, center: Vec2, color: tuple[int, int, int]) -> None:
        glow_size = config.CELL_SIZE * 3
        surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        gx = glow_size // 2
        gy = glow_size // 2

        pygame.draw.circle(surf, (*color, 40), (gx, gy), config.CELL_SIZE)
        pygame.draw.circle(surf, (*color, 100), (gx, gy), config.CELL_SIZE // 2)
        pygame.draw.circle(surf, (*color, 255), (gx, gy), max(2, config.CELL_SIZE // 4))

        self.screen.blit(surf, (center[0] - gx, center[1] - gy))

    def draw_world(self) -> None:
        self.screen.fill((8, 8, 10))

        radius = config.VIEW_SIZE // 2
        origin_x = self.player_pos[0] - radius
        origin_y = self.player_pos[1] - radius

        for vy in range(config.VIEW_SIZE):
            for vx in range(config.VIEW_SIZE):
                wx = origin_x + vx
                wy = origin_y + vy
                px = vx * config.CELL_SIZE
                py = vy * config.CELL_SIZE
                rect = pygame.Rect(px, py, config.CELL_SIZE, config.CELL_SIZE)

                if wx < 0 or wy < 0 or wx >= config.MAP_SIZE or wy >= config.MAP_SIZE:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                    continue

                if self.grid[wy][wx] == 1:
                    pygame.draw.rect(self.screen, (50, 54, 72), rect)
                else:
                    pygame.draw.rect(self.screen, (24, 30, 35), rect)

                pygame.draw.rect(self.screen, (18, 22, 28), rect, 1)

        for chest in self.chests:
            pos = self.world_to_screen(chest)
            if pos is None:
                continue
            x, y = pos
            rect = pygame.Rect(x + 4, y + 4, config.CELL_SIZE - 8, config.CELL_SIZE - 8)
            pygame.draw.rect(self.screen, (216, 173, 60), rect)

        for buff in self.buff_nodes:
            if not buff.active:
                continue
            pos = self.world_to_screen(buff.pos)
            if pos is None:
                continue
            x, y = pos
            rect = pygame.Rect(x + 6, y + 6, config.CELL_SIZE - 12, config.CELL_SIZE - 12)
            pygame.draw.rect(self.screen, (50, 208, 190), rect)

        for monster in self.active_monsters():
            pos = self.world_to_screen(monster.pos)
            if pos is None:
                continue
            center = (pos[0] + config.CELL_SIZE // 2, pos[1] + config.CELL_SIZE // 2)
            self.draw_glow_dot(center, (220, 75, 75))

        player_screen = self.world_to_screen(self.player_pos)
        if player_screen is not None:
            center = (
                player_screen[0] + config.CELL_SIZE // 2,
                player_screen[1] + config.CELL_SIZE // 2,
            )
            self.draw_glow_dot(center, (255, 245, 120))

    def draw_hud(self) -> None:
        hud_rect = pygame.Rect(0, self.view_pixels, self.view_pixels, self.hud_height)
        pygame.draw.rect(self.screen, (12, 14, 18), hud_rect)

        dash_left = 0.0 if not self.started else max(
            0.0, config.DASH_COOLDOWN_SECONDS - (self.elapsed_seconds - self.last_dash_used_at)
        )
        buff_left = max(0.0, self.player_buff_left)

        monster2_text = "spawned" if self.monster2 else f"in {max(0, int(config.MONSTER2_SPAWN_TIME_SECONDS - self.elapsed_seconds))}s"
        game_state = "waiting SPACE" if not self.started else "running"

        lines = [
            f"Score: {self.score:7.1f}   Time: {self.elapsed_seconds:6.1f}s   Chests left: {len(self.chests)}",
            f"State: {game_state}   Dash CD: {dash_left:5.1f}s   Buff left: {buff_left:5.1f}s",
            f"Map: {self.map_title}   Monster-2: {monster2_text}",
        ]

        y = self.view_pixels + 12
        for line in lines:
            img = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(img, (12, y))
            y += 30

    def draw_side_panel(self) -> None:
        panel_x = self.view_pixels
        panel_h = self.view_pixels + self.hud_height
        panel_rect = pygame.Rect(panel_x, 0, self.side_panel_width, panel_h)
        pygame.draw.rect(self.screen, (16, 20, 26), panel_rect)
        pygame.draw.line(self.screen, (36, 45, 58), (panel_x, 0), (panel_x, panel_h), 2)

        header = self.font.render("Radar + Chests", True, (230, 230, 230))
        self.screen.blit(header, (panel_x + 14, 14))

        y = 50
        sub = self.small_font.render("Off-screen monster hints", True, (195, 195, 195))
        self.screen.blit(sub, (panel_x + 14, y))
        y += 28

        monster_items: list[tuple[str, Monster | None]] = [("M1", self.monster1), ("M2", self.monster2)]
        for monster_name, monster in monster_items:
            if monster is None:
                line = self.small_font.render(f"{monster_name}: not spawned", True, (160, 160, 160))
                self.screen.blit(line, (panel_x + 14, y))
                y += 28
                continue

            if self.world_to_screen(monster.pos) is not None:
                line = self.small_font.render(f"{monster_name}: in view", True, (140, 208, 150))
                self.screen.blit(line, (panel_x + 14, y))
                y += 28
                continue

            direction = self.direction_to(monster.pos)
            dir_label = DIRECTION_LABELS.get(direction, "UNK")
            ex_mark = self.big_font.render("!", True, (238, 74, 74))
            self.screen.blit(ex_mark, (panel_x + 16, y - 14))
            self.draw_direction_arrow(
                self.screen,
                (panel_x + 80, y + 8),
                direction,
                (238, 74, 74),
                size=13,
            )
            line = self.small_font.render(f"{monster_name} approx: {dir_label}", True, (238, 174, 174))
            self.screen.blit(line, (panel_x + 102, y))
            y += 30

        y += 8
        nearest = self.nearest_chest()
        sub2 = self.small_font.render("Nearest chest", True, (210, 210, 210))
        self.screen.blit(sub2, (panel_x + 14, y))
        y += 28
        if nearest is None:
            line = self.small_font.render("No chest left", True, (160, 160, 160))
            self.screen.blit(line, (panel_x + 14, y))
            y += 30
        else:
            direction = self.direction_to(nearest)
            self.draw_direction_arrow(
                self.screen,
                (panel_x + 30, y + 8),
                direction,
                (247, 224, 74),
                size=13,
            )
            line = self.small_font.render(
                f"Nearest: ({nearest[0]}, {nearest[1]}) {DIRECTION_LABELS.get(direction, 'UNK')}",
                True,
                (247, 224, 74),
            )
            self.screen.blit(line, (panel_x + 52, y))
            y += 32

        y += 4
        sub3 = self.small_font.render("Chest coordinates", True, (210, 210, 210))
        self.screen.blit(sub3, (panel_x + 14, y))
        y += 26

        chest_items = sorted(self.chests)
        if not chest_items:
            line = self.small_font.render("None", True, (160, 160, 160))
            self.screen.blit(line, (panel_x + 14, y))
        else:
            max_rows = max(1, (panel_h - y - 12) // 22)
            for index, chest in enumerate(chest_items[:max_rows], start=1):
                is_nearest = chest == nearest
                color = (247, 224, 74) if is_nearest else (220, 220, 220)
                tail = "  <==" if is_nearest else ""
                text = self.small_font.render(
                    f"{index:02d}: ({chest[0]:3d}, {chest[1]:3d}){tail}",
                    True,
                    color,
                )
                self.screen.blit(text, (panel_x + 14, y))
                y += 22

    def draw_start_overlay(self) -> None:
        if self.started or self.finished:
            return

        overlay = pygame.Surface((self.view_pixels, self.view_pixels), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        text1 = self.big_font.render("READY", True, (245, 245, 245))
        text2 = self.font.render("Press SPACE to start", True, (230, 230, 230))
        text3 = self.font.render("Move: WASD/Arrow  Dash: SPACE", True, (210, 210, 210))

        self.screen.blit(text1, (self.view_pixels // 2 - text1.get_width() // 2, self.view_pixels // 2 - 65))
        self.screen.blit(text2, (self.view_pixels // 2 - text2.get_width() // 2, self.view_pixels // 2 - 10))
        self.screen.blit(text3, (self.view_pixels // 2 - text3.get_width() // 2, self.view_pixels // 2 + 25))

    def draw_end_overlay(self) -> None:
        if not self.finished:
            return

        overlay = pygame.Surface((self.view_pixels, self.view_pixels), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 130))
        self.screen.blit(overlay, (0, 0))

        title = "YOU WIN" if self.win else "GAME OVER"
        color = (130, 235, 140) if self.win else (240, 110, 110)

        text1 = self.big_font.render(title, True, color)
        text2 = self.font.render(f"Final score: {self.score:.1f}", True, (230, 230, 230))
        text3 = self.font.render("Press ESC to quit", True, (230, 230, 230))
        text4 = self.font.render(f"Map: {self.map_title}", True, (230, 230, 230))

        self.screen.blit(text1, (self.view_pixels // 2 - text1.get_width() // 2, self.view_pixels // 2 - 60))
        self.screen.blit(text2, (self.view_pixels // 2 - text2.get_width() // 2, self.view_pixels // 2 - 10))
        self.screen.blit(text3, (self.view_pixels // 2 - text3.get_width() // 2, self.view_pixels // 2 + 25))
        self.screen.blit(text4, (self.view_pixels // 2 - text4.get_width() // 2, self.view_pixels // 2 + 55))

        if self.replay_folder is not None:
            replay_line = self.font.render(f"Replay: {self.replay_folder.name}", True, (200, 200, 200))
            self.screen.blit(replay_line, (self.view_pixels // 2 - replay_line.get_width() // 2, self.view_pixels // 2 + 85))

    def draw(self) -> None:
        self.draw_world()
        self.draw_hud()
        self.draw_side_panel()
        self.draw_start_overlay()
        self.draw_end_overlay()
        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            dt = min(0.05, self.clock.tick(config.FPS) / 1000.0)
            running = self.handle_events()
            self.update(dt)
            self.draw()

        pygame.quit()


def main() -> None:
    game = TreasureHideGame()
    game.run()


if __name__ == "__main__":
    main()
