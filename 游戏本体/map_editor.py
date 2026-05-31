from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pygame
except ImportError:
    print("需要 pygame：请先执行 pip install pygame")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None

MAP_SIZE = 128
WINDOW_WIDTH = 1180
WINDOW_HEIGHT = 900
CANVAS_MARGIN = 16
SIDE_PANEL_WIDTH = 300

GRID_BG = (22, 28, 34)
GRID_LINE = (36, 44, 52)
ROAD_COLOR = (34, 40, 48)
WALL_COLOR = (230, 230, 230)
TEXT_COLOR = (230, 230, 230)
ACCENT = (255, 211, 67)
WARN = (235, 106, 106)


def load_zh_font(size: int, bold: bool = False) -> pygame.font.Font:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    for name in candidates:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)

    # 兜底
    return pygame.font.SysFont("simhei", size, bold=bold)


def default_map_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "地图" / "tensorImg"


def parse_text_rows(text: str) -> list[list[int]] | None:
    rows: list[list[int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if set(line).issubset({"0", "1"}):
            row = [int(ch) for ch in line]
        else:
            tokens = [tok for tok in re.split(r"[^01]+", line) if tok]
            if not tokens:
                continue
            row = [int(tok) for tok in tokens]
        rows.append(row)

    if not rows:
        return None

    width = len(rows[0])
    if width == 0 or any(len(r) != width for r in rows):
        return None
    return rows


def matrix_from_json(value: object) -> list[list[int]] | None:
    if not isinstance(value, list) or not value:
        return None
    matrix: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or not row:
            return None
        out: list[int] = []
        for cell in row:
            try:
                v = int(cell)
            except Exception:
                return None
            if v not in (0, 1):
                return None
            out.append(v)
        matrix.append(out)

    width = len(matrix[0])
    if width == 0 or any(len(r) != width for r in matrix):
        return None
    return matrix


def load_matrix(path: Path) -> list[list[int]]:
    suffix = path.suffix.lower()

    if suffix == ".npy" and np is not None:
        arr = np.load(path, allow_pickle=False)
        while getattr(arr, "ndim", 0) > 2:
            arr = arr[0]
        if getattr(arr, "ndim", 0) != 2:
            raise ValueError("NPY 文件必须是二维矩阵")
        matrix = [[int(v) for v in row] for row in arr.tolist()]
    elif suffix == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        matrix = matrix_from_json(parsed) or []
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gbk")
        matrix = parse_text_rows(text) or []

    if not matrix:
        raise ValueError(f"无法解析地图文件: {path.name}")

    h = len(matrix)
    w = len(matrix[0])
    for y in range(h):
        for x in range(w):
            if matrix[y][x] not in (0, 1):
                raise ValueError(f"地图里出现非0/1值: ({x},{y})")

    # 统一裁剪/填充到 128x128
    out = [[0 for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
    for y in range(min(MAP_SIZE, h)):
        for x in range(min(MAP_SIZE, w)):
            out[y][x] = 1 if matrix[y][x] == 1 else 0

    # 边界设为障碍，避免坏地图
    for i in range(MAP_SIZE):
        out[0][i] = 1
        out[MAP_SIZE - 1][i] = 1
        out[i][0] = 1
        out[i][MAP_SIZE - 1] = 1

    return out


def save_matrix_csv(path: Path, grid: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(",".join(str(cell) for cell in row) for row in grid)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def list_map_files(map_dir: Path) -> list[Path]:
    map_dir.mkdir(parents=True, exist_ok=True)
    exts = {".csv", ".txt", ".json", ".npy"}
    return sorted([p for p in map_dir.iterdir() if p.is_file() and p.suffix.lower() in exts], key=lambda p: p.name)


def choose_start_mode(map_dir: Path, new_name: str | None) -> tuple[list[list[int]], Path | None, bool]:
    files = list_map_files(map_dir)
    print("=" * 58)
    print(f"地图工作目录: {map_dir}")
    print("1) 编辑已有地图")
    print("2) 创建新地图")
    print("=" * 58)

    while True:
        raw = input("请选择模式(1/2): ").strip()
        if raw in {"1", "2"}:
            break

    if raw == "1":
        if not files:
            print("地图目录中没有可编辑文件，自动切换为新建地图。")
        else:
            print("可编辑地图列表:")
            for idx, f in enumerate(files, start=1):
                print(f"  {idx:02d}. {f.name}")
            while True:
                pick = input(f"输入编号(1-{len(files)}): ").strip()
                try:
                    n = int(pick)
                except Exception:
                    continue
                if 1 <= n <= len(files):
                    path = files[n - 1]
                    grid = load_matrix(path)
                    return grid, path, False

    # 新建地图
    grid = [[0 for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
    for i in range(MAP_SIZE):
        grid[0][i] = 1
        grid[MAP_SIZE - 1][i] = 1
        grid[i][0] = 1
        grid[i][MAP_SIZE - 1] = 1

    file_name = (new_name or "new_map.csv").strip()
    if not file_name:
        file_name = "new_map.csv"
    if not Path(file_name).suffix:
        file_name += ".csv"

    target_path = map_dir / Path(file_name).name
    print(f"新建地图目标文件名: {target_path.name}")
    print("注意: 在你确认保存前，不会写入地图目录。")
    return grid, target_path, True


class MapEditor:
    def __init__(self, map_dir: Path, grid: list[list[int]], target_path: Path | None, is_new_map: bool) -> None:
        self.map_dir = map_dir
        self.grid = [row[:] for row in grid]
        self.target_path = target_path
        self.is_new_map = is_new_map
        self.unsaved = False

        self.canvas_width = WINDOW_WIDTH - SIDE_PANEL_WIDTH - CANVAS_MARGIN * 2
        self.canvas_height = WINDOW_HEIGHT - CANVAS_MARGIN * 2

        self.cell_size = 6
        self.camera_x = 0
        self.camera_y = 0

        self.is_painting = False
        self.paint_value = 1
        self.brush_size = 1
        self.pending_save_confirm = False
        self.status_line = ""
        self.brush_minus_button = pygame.Rect(0, 0, 0, 0)
        self.brush_plus_button = pygame.Rect(0, 0, 0, 0)

        pygame.init()
        pygame.display.set_caption("地图编辑器（0/1）")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = load_zh_font(20, bold=True)
        self.small_font = load_zh_font(16)

    def visible_cols(self) -> int:
        return max(1, self.canvas_width // self.cell_size)

    def visible_rows(self) -> int:
        return max(1, self.canvas_height // self.cell_size)

    def clamp_camera(self) -> None:
        self.camera_x = max(0, min(MAP_SIZE - self.visible_cols(), self.camera_x))
        self.camera_y = max(0, min(MAP_SIZE - self.visible_rows(), self.camera_y))

    def screen_to_cell(self, mx: int, my: int) -> tuple[int, int] | None:
        left = CANVAS_MARGIN
        top = CANVAS_MARGIN
        if mx < left or my < top:
            return None
        if mx >= left + self.canvas_width or my >= top + self.canvas_height:
            return None

        vx = (mx - left) // self.cell_size
        vy = (my - top) // self.cell_size

        gx = self.camera_x + vx
        gy = self.camera_y + vy
        if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
            return int(gx), int(gy)
        return None

    def paint_at_mouse(self) -> None:
        mx, my = pygame.mouse.get_pos()
        cell = self.screen_to_cell(mx, my)
        if cell is None:
            return
        cx, cy = cell
        radius = max(0, self.brush_size - 1)
        changed = False

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = cx + dx
                y = cy + dy
                if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
                    continue
                if self.grid[y][x] != self.paint_value:
                    self.grid[y][x] = self.paint_value
                    changed = True

        if changed:
            self.unsaved = True

    def pan(self, dx: int, dy: int) -> None:
        self.camera_x += dx
        self.camera_y += dy
        self.clamp_camera()

    def zoom(self, delta: int) -> None:
        old = self.cell_size
        self.cell_size = max(3, min(20, self.cell_size + delta))
        if self.cell_size != old:
            self.clamp_camera()

    def do_save(self) -> None:
        if self.target_path is None:
            self.status_line = "未设置目标文件名，无法保存"
            return

        if self.target_path.exists() and self.is_new_map:
            # 新建时如果文件已存在，避免无提示覆盖
            self.status_line = f"目标文件已存在: {self.target_path.name}，请重启并换文件名"
            return

        save_matrix_csv(self.target_path, self.grid)
        self.unsaved = False
        self.is_new_map = False
        self.pending_save_confirm = False
        self.status_line = f"已保存: {self.target_path.name}"

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if event.button == 1 and self.brush_minus_button.collidepoint(mx, my):
                self.brush_size = max(1, self.brush_size - 1)
                self.status_line = f"画笔粗细: {self.brush_size}"
                return True
            if event.button == 1 and self.brush_plus_button.collidepoint(mx, my):
                self.brush_size = min(12, self.brush_size + 1)
                self.status_line = f"画笔粗细: {self.brush_size}"
                return True

            if event.button == 1:
                self.is_painting = True
                self.paint_value = 1
                self.paint_at_mouse()
            elif event.button == 3:
                self.is_painting = True
                self.paint_value = 0
                self.paint_at_mouse()
            elif event.button == 4:
                self.zoom(1)
            elif event.button == 5:
                self.zoom(-1)

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button in (1, 3):
                self.is_painting = False

        if event.type == pygame.KEYDOWN:
            ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL

            if event.key == pygame.K_ESCAPE:
                return False

            if event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                self.zoom(1)
            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                self.zoom(-1)
            elif event.key == pygame.K_UP:
                self.pan(0, -4)
            elif event.key == pygame.K_DOWN:
                self.pan(0, 4)
            elif event.key == pygame.K_LEFT:
                self.pan(-4, 0)
            elif event.key == pygame.K_RIGHT:
                self.pan(4, 0)
            elif event.key == pygame.K_1:
                self.paint_value = 1
            elif event.key == pygame.K_0:
                self.paint_value = 0
            elif event.key == pygame.K_LEFTBRACKET:
                self.brush_size = max(1, self.brush_size - 1)
                self.status_line = f"画笔粗细: {self.brush_size}"
            elif event.key == pygame.K_RIGHTBRACKET:
                self.brush_size = min(12, self.brush_size + 1)
                self.status_line = f"画笔粗细: {self.brush_size}"
            elif event.key == pygame.K_c:
                for y in range(1, MAP_SIZE - 1):
                    for x in range(1, MAP_SIZE - 1):
                        self.grid[y][x] = 0
                self.unsaved = True
                self.status_line = "已清空内部区域"
            elif event.key == pygame.K_s and ctrl:
                self.pending_save_confirm = True
                if self.is_new_map:
                    self.status_line = f"确认保存新地图到 {self.target_path.name}? 按 Y 保存 / N 取消"
                else:
                    self.status_line = f"确认覆盖保存 {self.target_path.name}? 按 Y 保存 / N 取消"
            elif self.pending_save_confirm and event.key == pygame.K_y:
                self.do_save()
            elif self.pending_save_confirm and event.key == pygame.K_n:
                self.pending_save_confirm = False
                self.status_line = "已取消保存"

        return True

    def draw_grid(self) -> None:
        left = CANVAS_MARGIN
        top = CANVAS_MARGIN
        canvas = pygame.Rect(left, top, self.canvas_width, self.canvas_height)
        pygame.draw.rect(self.screen, GRID_BG, canvas)

        cols = self.visible_cols()
        rows = self.visible_rows()

        for vy in range(rows):
            gy = self.camera_y + vy
            if gy >= MAP_SIZE:
                break
            py = top + vy * self.cell_size
            for vx in range(cols):
                gx = self.camera_x + vx
                if gx >= MAP_SIZE:
                    break
                px = left + vx * self.cell_size
                rect = pygame.Rect(px, py, self.cell_size, self.cell_size)
                color = WALL_COLOR if self.grid[gy][gx] == 1 else ROAD_COLOR
                pygame.draw.rect(self.screen, color, rect)
                if self.cell_size >= 6:
                    pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

        pygame.draw.rect(self.screen, (70, 85, 100), canvas, 2)

    def draw_panel(self) -> None:
        panel_x = WINDOW_WIDTH - SIDE_PANEL_WIDTH
        panel_rect = pygame.Rect(panel_x, 0, SIDE_PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, (14, 19, 24), panel_rect)
        pygame.draw.line(self.screen, (45, 54, 66), (panel_x, 0), (panel_x, WINDOW_HEIGHT), 2)

        map_name = self.target_path.name if self.target_path else "(未命名)"
        mode_text = "新建地图" if self.is_new_map else "编辑已有"
        dirty_text = "未保存" if self.unsaved else "已保存"
        dirty_color = WARN if self.unsaved else (130, 210, 130)

        lines = [
            (self.font, "地图编辑器", ACCENT),
            (self.small_font, f"工作目录: {self.map_dir}", TEXT_COLOR),
            (self.small_font, f"模式: {mode_text}", TEXT_COLOR),
            (self.small_font, f"文件: {map_name}", TEXT_COLOR),
            (self.small_font, f"状态: {dirty_text}", dirty_color),
            (self.small_font, f"画笔值: {self.paint_value}", TEXT_COLOR),
            (self.small_font, f"画笔粗细: {self.brush_size}", TEXT_COLOR),
            (self.small_font, f"视图: ({self.camera_x},{self.camera_y}) zoom={self.cell_size}", TEXT_COLOR),
            (self.small_font, "", TEXT_COLOR),
            (self.small_font, "操作说明:", ACCENT),
            (self.small_font, "左键=画1(障碍) 右键=画0(道路)", TEXT_COLOR),
            (self.small_font, "Ctrl+S=请求保存(需Y确认)", TEXT_COLOR),
            (self.small_font, "Y/N=确认或取消保存", TEXT_COLOR),
            (self.small_font, "方向键=平移  +/-或滚轮=缩放", TEXT_COLOR),
            (self.small_font, "1/0=切换画笔值  [ / ]=画笔粗细", TEXT_COLOR),
            (self.small_font, "C=清空内部区域", TEXT_COLOR),
            (self.small_font, "ESC=退出", TEXT_COLOR),
        ]

        y = 18
        for font_obj, text, color in lines:
            if text:
                image = font_obj.render(text, True, color)
                self.screen.blit(image, (panel_x + 12, y))
                y += image.get_height() + 7
            else:
                y += 8

        # 画笔粗细按钮（+ / -）
        btn_y = 210
        btn_w = 40
        btn_h = 32
        self.brush_minus_button = pygame.Rect(panel_x + 190, btn_y, btn_w, btn_h)
        self.brush_plus_button = pygame.Rect(panel_x + 240, btn_y, btn_w, btn_h)

        pygame.draw.rect(self.screen, (52, 62, 76), self.brush_minus_button, border_radius=6)
        pygame.draw.rect(self.screen, (52, 62, 76), self.brush_plus_button, border_radius=6)
        pygame.draw.rect(self.screen, (95, 110, 132), self.brush_minus_button, 1, border_radius=6)
        pygame.draw.rect(self.screen, (95, 110, 132), self.brush_plus_button, 1, border_radius=6)

        minus_img = self.font.render("-", True, TEXT_COLOR)
        plus_img = self.font.render("+", True, TEXT_COLOR)
        self.screen.blit(
            minus_img,
            (
                self.brush_minus_button.centerx - minus_img.get_width() // 2,
                self.brush_minus_button.centery - minus_img.get_height() // 2 - 1,
            ),
        )
        self.screen.blit(
            plus_img,
            (
                self.brush_plus_button.centerx - plus_img.get_width() // 2,
                self.brush_plus_button.centery - plus_img.get_height() // 2 - 1,
            ),
        )

        if self.status_line:
            status_img = self.small_font.render(self.status_line, True, ACCENT)
            self.screen.blit(status_img, (CANVAS_MARGIN, WINDOW_HEIGHT - 28))

        if self.pending_save_confirm:
            overlay = pygame.Surface((self.canvas_width, 42), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (CANVAS_MARGIN, 0))
            tip = "保存确认中: 按 Y 保存 / N 取消"
            tip_img = self.font.render(tip, True, ACCENT)
            self.screen.blit(tip_img, (CANVAS_MARGIN + 10, 8))

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60)
            _ = dt

            for event in pygame.event.get():
                running = self.handle_event(event)
                if not running:
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.pan(0, -2)
            if keys[pygame.K_s] and not (pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.pan(0, 2)
            if keys[pygame.K_a]:
                self.pan(-2, 0)
            if keys[pygame.K_d]:
                self.pan(2, 0)

            if self.is_painting:
                self.paint_at_mouse()

            self.screen.fill((8, 10, 12))
            self.draw_grid()
            self.draw_panel()
            pygame.display.flip()

        pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description="地图编辑器（0/1，工作路径=地图目录）")
    parser.add_argument("--map-dir", type=Path, default=default_map_dir(), help="地图目录，默认 ../地图/tensorImg")
    parser.add_argument("--new-name", default=None, help="新建地图默认文件名，例如 custom_01.csv")
    args = parser.parse_args()

    map_dir = args.map_dir.resolve()
    map_dir.mkdir(parents=True, exist_ok=True)

    try:
        grid, target_path, is_new = choose_start_mode(map_dir, args.new_name)
    except Exception as exc:
        print(f"初始化编辑器失败: {exc}")
        return 1

    editor = MapEditor(map_dir=map_dir, grid=grid, target_path=target_path, is_new_map=is_new)
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
