from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pygame
except ImportError:
    print("需要 pygame：请先执行 pip install pygame")
    sys.exit(1)


def default_replay_root() -> Path:
    return Path(__file__).resolve().parent.parent / "录像文件"


def find_latest_replay_folder(replay_root: Path) -> Path | None:
    folders = [p for p in replay_root.iterdir() if p.is_dir()]
    if not folders:
        return None
    return max(folders, key=lambda p: p.name)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: int(r.get("time", 0)))
    return records


def create_map_surface(grid: list[list[int]], cell_size: int) -> pygame.Surface:
    size = len(grid)
    pixels = size * cell_size
    surface = pygame.Surface((pixels, pixels))

    road_color = (24, 30, 35)
    wall_color = (50, 54, 72)
    grid_color = (18, 22, 28)

    for y in range(size):
        for x in range(size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(surface, wall_color if grid[y][x] == 1 else road_color, rect)
            if cell_size >= 5:
                pygame.draw.rect(surface, grid_color, rect, 1)

    return surface


def draw_glow_dot(surface: pygame.Surface, center: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
    glow_size = max(10, radius * 5)
    glow = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
    gx = glow_size // 2
    gy = glow_size // 2

    pygame.draw.circle(glow, (*color, 45), (gx, gy), max(3, radius * 2))
    pygame.draw.circle(glow, (*color, 120), (gx, gy), max(2, int(radius * 1.2)))
    pygame.draw.circle(glow, (*color, 255), (gx, gy), max(2, radius))

    surface.blit(glow, (center[0] - gx, center[1] - gy))


def world_to_screen(pos: list[int] | tuple[int, int], cell_size: int) -> tuple[int, int]:
    return int(pos[0]) * cell_size + cell_size // 2, int(pos[1]) * cell_size + cell_size // 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="播放录像文件（全局视野）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--replay", type=Path, default=None, help="指定录像文件夹路径，不传则默认播放最新一局")
    parser.add_argument("--replay-root", type=Path, default=default_replay_root(), help="录像根目录")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument("--fps", type=int, default=60, help="渲染帧率")
    parser.add_argument("--cell-size", type=int, default=0, help="每格像素，0表示自动")
    args = parser.parse_args()

    replay_root = args.replay_root.resolve()
    if not replay_root.exists() or not replay_root.is_dir():
        print(f"录像根目录不存在: {replay_root}")
        return 1

    replay_dir = args.replay.resolve() if args.replay else find_latest_replay_folder(replay_root)
    if replay_dir is None or (not replay_dir.exists()) or (not replay_dir.is_dir()):
        print("没有可播放的录像文件夹。")
        return 1

    summary_path = replay_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    map_file = summary.get("map_file", "map.json")
    map_path = replay_dir / map_file
    if not map_path.exists():
        print(f"录像缺少地图文件: {map_path}")
        return 1

    map_payload = load_json(map_path)
    grid = map_payload.get("grid")
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        print("地图文件格式错误。")
        return 1

    map_size = int(map_payload.get("map_size", len(grid)))
    if len(grid) != map_size:
        map_size = len(grid)

    record_file = summary.get("record_file", "record.jsonl")
    record_path = replay_dir / record_file
    if not record_path.exists():
        print(f"录像缺少逐秒记录文件: {record_path}")
        return 1

    records = load_records(record_path)
    if not records:
        print("逐秒记录为空，无法播放。")
        return 1

    final_score = float(summary.get("final_score", records[-1].get("score", 0.0)))
    map_title = str(summary.get("map_title", summary.get("map_source", "Unknown Map")))

    pygame.init()
    pygame.display.set_caption(f"录像播放 - {replay_dir.name}")

    if args.cell_size > 0:
        cell_size = args.cell_size
    else:
        max_pixels = 860
        cell_size = max(3, min(9, max_pixels // map_size))

    map_pixels = map_size * cell_size
    side_width = 340
    width = map_pixels + side_width
    height = max(map_pixels, 430)

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 20)
    small_font = pygame.font.SysFont("consolas", 17)
    big_font = pygame.font.SysFont("consolas", 28)

    map_surface = create_map_surface(grid, cell_size)

    frame_index = 0
    play_speed = max(0.1, args.speed)
    playing = True
    time_accum = 0.0

    running = True
    while running:
        dt = min(0.08, clock.tick(max(30, args.fps)) / 1000.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_LEFT:
                    frame_index = max(0, frame_index - 1)
                    playing = False
                    time_accum = 0.0
                elif event.key == pygame.K_RIGHT:
                    frame_index = min(len(records) - 1, frame_index + 1)
                    playing = False
                    time_accum = 0.0
                elif event.key == pygame.K_HOME:
                    frame_index = 0
                    playing = False
                    time_accum = 0.0
                elif event.key == pygame.K_END:
                    frame_index = len(records) - 1
                    playing = False
                    time_accum = 0.0
                elif event.key == pygame.K_r:
                    frame_index = 0
                    playing = True
                    time_accum = 0.0
                elif event.key == pygame.K_UP:
                    play_speed = min(16.0, play_speed * 2.0)
                elif event.key == pygame.K_DOWN:
                    play_speed = max(0.25, play_speed / 2.0)

        if playing and frame_index < len(records) - 1:
            time_accum += dt * play_speed
            while time_accum >= 1.0 and frame_index < len(records) - 1:
                frame_index += 1
                time_accum -= 1.0

        frame = records[frame_index]

        screen.fill((10, 12, 14))
        screen.blit(map_surface, (0, 0))

        for chest in frame.get("chests", []):
            if not isinstance(chest, list) or len(chest) != 2:
                continue
            rect = pygame.Rect(
                int(chest[0]) * cell_size + max(1, cell_size // 5),
                int(chest[1]) * cell_size + max(1, cell_size // 5),
                max(2, cell_size - 2 * max(1, cell_size // 5)),
                max(2, cell_size - 2 * max(1, cell_size // 5)),
            )
            pygame.draw.rect(screen, (216, 173, 60), rect)

        for buff in frame.get("buffs", []):
            if not isinstance(buff, dict):
                continue
            if not buff.get("active", False):
                continue
            pos = buff.get("pos")
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            rect = pygame.Rect(
                int(pos[0]) * cell_size + max(1, cell_size // 4),
                int(pos[1]) * cell_size + max(1, cell_size // 4),
                max(2, cell_size - 2 * max(1, cell_size // 4)),
                max(2, cell_size - 2 * max(1, cell_size // 4)),
            )
            pygame.draw.rect(screen, (50, 208, 190), rect)

        hero = frame.get("hero")
        if isinstance(hero, list) and len(hero) == 2:
            center = world_to_screen(hero, cell_size)
            draw_glow_dot(screen, center, (255, 245, 120), max(2, cell_size // 3))

        monster1 = frame.get("monster1")
        if isinstance(monster1, list) and len(monster1) == 2:
            center = world_to_screen(monster1, cell_size)
            draw_glow_dot(screen, center, (220, 75, 75), max(2, cell_size // 3))

        monster2 = frame.get("monster2")
        if isinstance(monster2, list) and len(monster2) == 2:
            center = world_to_screen(monster2, cell_size)
            draw_glow_dot(screen, center, (240, 95, 95), max(2, cell_size // 3))

        panel_x = map_pixels
        pygame.draw.rect(screen, (16, 20, 26), pygame.Rect(panel_x, 0, side_width, height))
        pygame.draw.line(screen, (40, 48, 62), (panel_x, 0), (panel_x, height), 2)

        status = "Playing" if playing else "Paused"
        current_time = int(frame.get("time", frame_index))
        current_score = float(frame.get("score", 0.0))

        text_lines = [
            (big_font, f"REPLAY ({status})", (240, 240, 240)),
            (font, f"Folder: {replay_dir.name}", (210, 210, 210)),
            (font, f"Map: {map_title}", (210, 210, 210)),
            (font, f"Time: {current_time}s / {int(records[-1].get('time', len(records)-1))}s", (230, 230, 230)),
            (font, f"Score now: {current_score:.1f}", (247, 224, 74)),
            (font, f"Final score: {final_score:.1f}", (247, 224, 74)),
            (font, f"Speed: x{play_speed:.2f}", (200, 220, 255)),
            (small_font, "GLOBAL VIEW (全局视野)", (170, 200, 170)),
            (small_font, "Space: play/pause", (180, 180, 180)),
            (small_font, "Left/Right: prev/next second", (180, 180, 180)),
            (small_font, "Up/Down: speed +/-", (180, 180, 180)),
            (small_font, "Home/End: start/end", (180, 180, 180)),
            (small_font, "R: restart", (180, 180, 180)),
            (small_font, "Esc: quit", (180, 180, 180)),
        ]

        y = 16
        for font_obj, text, color in text_lines:
            rendered = font_obj.render(text, True, color)
            screen.blit(rendered, (panel_x + 14, y))
            y += rendered.get_height() + 8

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
