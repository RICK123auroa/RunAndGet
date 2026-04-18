from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def parse_duration(duration_text: str) -> float:
    text = duration_text.strip().lower().replace(" ", "")
    if not text:
        raise ValueError("empty duration")

    pattern = re.compile(r"(\d+(?:\.\d+)?)([smhdw])")
    matches = list(pattern.finditer(text))
    if not matches or "".join(m.group(0) for m in matches) != text:
        raise ValueError("duration format should be like 8h, 90m, 1h30m")

    seconds = 0.0
    for match in matches:
        value = float(match.group(1))
        unit = match.group(2)
        seconds += value * UNIT_SECONDS[unit]

    if seconds <= 0:
        raise ValueError("duration must be greater than zero")
    return seconds


def default_replay_root() -> Path:
    return Path(__file__).resolve().parent.parent / "录像文件"


def calc_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def list_expired_entries(root: Path, older_than_seconds: float) -> list[Path]:
    now = time.time()
    targets: list[Path] = []

    for item in sorted(root.iterdir(), key=lambda p: p.name):
        if not item.is_dir():
            continue
        age_seconds = now - item.stat().st_mtime
        if age_seconds >= older_than_seconds:
            targets.append(item)

    return targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="清理录像目录中超过指定时长的录像文件夹。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--older-than",
        default="8h",
        help="清理早于该时长的录像，可写 8h / 90m / 1h30m / 2d 等",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=default_replay_root(),
        help="录像根目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将被删除的目录，不实际删除",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="不询问确认，直接执行",
    )

    args = parser.parse_args()

    try:
        older_than_seconds = parse_duration(args.older_than)
    except ValueError as exc:
        print(f"时间参数无效: {exc}")
        return 2

    replay_root = args.replay_root.resolve()
    if not replay_root.exists() or not replay_root.is_dir():
        print(f"录像目录不存在: {replay_root}")
        return 1

    targets = list_expired_entries(replay_root, older_than_seconds)
    if not targets:
        print(f"没有需要清理的录像。目录: {replay_root}")
        return 0

    total_bytes = sum(calc_size_bytes(path) for path in targets)

    print(f"录像目录: {replay_root}")
    print(f"清理条件: 早于 {args.older_than}")
    print(f"匹配数量: {len(targets)}")
    print(f"预计释放: {format_size(total_bytes)}")
    print("将处理以下目录:")
    for path in targets:
        print(f"- {path.name}")

    if args.dry_run:
        print("dry-run 模式，不会实际删除。")
        return 0

    if not args.force:
        answer = input("确认删除以上目录？输入 y 继续: ").strip().lower()
        if answer != "y":
            print("已取消。")
            return 0

    removed = 0
    for path in targets:
        shutil.rmtree(path, ignore_errors=False)
        removed += 1

    print(f"清理完成，已删除 {removed} 个目录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
