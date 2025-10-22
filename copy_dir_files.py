#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按规则从 src 下每个一级子文件夹复制对应的二进制文件到 dst 的对应子文件夹。

规则（自动判定子文件夹类型）：
- 子文件夹名以 coreutils 开头  -> 复制名为 "dir" 的文件
- 子文件夹名以 diffutils 开头 -> 复制名为 "sdiff" 的文件
- 子文件夹名以 findutils 开头 -> 复制名为 "find" 的文件
- 子文件夹名包含 gmp         -> 复制名为 "libgmp.so" 的文件
- 子文件夹名以 putty 开头    -> 复制名为 "plink" 的文件

用法示例：
python3 copy_specific_bins.py \
    --src /home/hww/lhf/SigmaDiff-main/data/binaries-2025/binaries \
    --dst /home/hww/lhf/SigmaDiff-main/data/binaries_test_all \
    --recursive --copy-all

参数说明:
--src       源目录（包含多个以库名/版本命名的子文件夹）
--dst       目标目录（在此创建对应子文件夹并放置复制的文件）
--recursive 递归查找（在子文件夹中递归查找匹配文件）
--copy-all  如果找到多个匹配，复制全部；否则只复制第一个
--overwrite 如果目标文件已存在，允许覆盖（默认不覆盖）
--dry-run   仅打印将要做的操作，不实际复制
"""

import argparse
import os
from pathlib import Path
import shutil
import sys

# 映射函数：根据子文件夹名返回要查找的文件名（或 None 表示跳过）
def target_name_for_subdir(subdir_name: str):
    s = subdir_name.lower()
    # 优先按 startswith 判断（版本号后缀等不会影响）
    if s.startswith("coreutils"):
        return "dir"
    if s.startswith("diffutils"):
        return "sdiff"
    if s.startswith("findutils"):
        return "find"
    if "gmp" in s:
        return "libgmp.so"
    if s.startswith("putty"):
        return "plink"
    return None

def find_matches_in_dir(src_subdir: Path, target_file_name: str, recursive: bool):
    """返回在 src_subdir 中匹配 target_file_name 的文件路径列表"""
    matches = []
    if recursive:
        for root, _, files in os.walk(src_subdir):
            for f in files:
                if f == target_file_name:
                    matches.append(Path(root) / f)
    else:
        p = src_subdir / target_file_name
        if p.is_file():
            matches.append(p)
    return matches

def main():
    parser = argparse.ArgumentParser(description="按库类型复制特定文件到目标对应子文件夹")
    parser.add_argument("--src", required=True, help="源目录（包含很多子文件夹）")
    parser.add_argument("--dst", required=True, help="目标目录（将创建同名子文件夹并放入文件）")
    parser.add_argument("--recursive", action="store_true", help="在每个子文件夹中递归查找匹配文件（默认只查顶层）")
    parser.add_argument("--copy-all", action="store_true", help="如果找到多个匹配文件，复制全部；否则只复制第一个")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖目标已存在的同名文件（默认跳过）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要做的操作，不实际复制")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    recursive = args.recursive
    copy_all = args.copy_all
    overwrite = args.overwrite
    dry_run = args.dry_run

    if not src.is_dir():
        print(f"[ERROR] 源目录不存在或不是目录: {src}", file=sys.stderr)
        sys.exit(1)
    dst.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_dirs": 0,
        "skipped_dirs": 0,
        "no_rule": 0,
        "miss": 0,
        "copied": 0,
        "errors": 0,
        "already_exists_skipped": 0
    }

    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue
        stats["total_dirs"] += 1
        subname = entry.name
        target_name = target_name_for_subdir(subname)
        if target_name is None:
            # 不是感兴趣的第三方库，跳过
            print(f"[SKIP] 子文件夹不在规则内，跳过: {subname}")
            stats["no_rule"] += 1
            continue

        dst_sub = dst / subname
        dst_sub.mkdir(parents=True, exist_ok=True)

        matches = find_matches_in_dir(entry, target_name, recursive)
        if not matches:
            print(f"[MISS] 在 {subname} 中未找到 '{target_name}'")
            stats["miss"] += 1
            continue

        to_copy = matches if copy_all else [matches[0]]
        for src_file in to_copy:
            dst_file = dst_sub / src_file.name
            if dst_file.exists() and not overwrite:
                print(f"[EXISTS] 目标已存在且未允许覆盖，跳过: {dst_file}")
                stats["already_exists_skipped"] += 1
                continue

            if dry_run:
                print(f"[DRY-RUN] 将复制: {src_file} -> {dst_file}")
                stats["copied"] += 1
            else:
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"[OK] 复制: {src_file} -> {dst_file}")
                    stats["copied"] += 1
                except Exception as e:
                    print(f"[ERROR] 复制失败: {src_file} -> {dst_file} ; err: {e}")
                    stats["errors"] += 1

    print("\n==== 汇总 ====")
    print(f"遍历子文件夹总数: {stats['total_dirs']}")
    print(f"不在规则内的文件夹数: {stats['no_rule']}")
    print(f"未找到目标文件的文件夹数: {stats['miss']}")
    print(f"已复制文件数 (包含 dry-run 计数): {stats['copied']}")
    print(f"已跳过已存在文件数: {stats['already_exists_skipped']}")
    print(f"复制错误数: {stats['errors']}")
    if dry_run:
        print("注意: 使用了 --dry-run，未实际复制任何文件。")

if __name__ == "__main__":
    main()
