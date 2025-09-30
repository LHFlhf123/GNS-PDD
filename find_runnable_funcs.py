#!/usr/bin/env python3
"""
find_runnable_funcs.py

用法举例：
  python find_runnable_funcs.py --dir /path/to/coreutils-5.93-O0_dirstripped
  python find_runnable_funcs.py --dir /path/.../coreutils-5.93-O0_dirstripped \
                                --other_dir /path/.../coreutils-5.93-O3_dirstripped

输出会打印并写入 ./funcs_with_nodes.csv
"""
import argparse
import os
import re
import csv

FUN_HEADER_RE = re.compile(r"^#FUN_([0-9A-Fa-f]+)")

def parse_nodelabel(file_path):
    """
    Returns dict: { 'FUN_00102760': node_count, ... }
    """
    res = {}
    cur_fun = None
    cur_count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            m = FUN_HEADER_RE.match(line)
            if m:
                # flush previous
                if cur_fun is not None:
                    res[cur_fun] = res.get(cur_fun, 0) + cur_count
                cur_fun = "FUN_" + m.group(1)
                cur_count = 0
            else:
                # typical node line starts with digits or digit|..., but we'll count any non-header non-empty line
                if cur_fun is not None and line.strip() and not line.strip().startswith('#'):
                    # treat as a node line
                    cur_count += 1
        # flush last
        if cur_fun is not None:
            res[cur_fun] = res.get(cur_fun, 0) + cur_count
    return res

def parse_addr2funcname(file_path):
    """
    Returns dict mapping decimal_address(int) -> funcname (string)
    Accepts lines like "1059472, getpwnam_thunk" or "0x00102760, FUN_00102760"
    """
    out = {}
    if not os.path.exists(file_path):
        return out
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split on comma
            parts = [p.strip() for p in line.split(',', 1)]
            if len(parts) == 0:
                continue
            addr_token = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            try:
                addr_int = int(addr_token, 0)  # accepts hex (0x..) or decimal
            except Exception:
                # try remove non-digits and parse decimal
                s = re.sub(r"[^\d]", "", addr_token)
                if s:
                    addr_int = int(s)
                else:
                    continue
            out[addr_int] = name
    return out

def fun_hex_to_decimal(fun_label):
    # fun_label like "FUN_00102760" -> take hex part
    if not fun_label.startswith("FUN_"):
        return None
    hexpart = fun_label.split("FUN_")[-1]
    try:
        return int(hexpart, 16)
    except Exception:
        return None

def find_files_in_dir(d):
    # find single *_nodelabel.txt and *_addr2funcname.txt in directory (or multiple)
    nodelabels = []
    addrfiles = []
    for fname in os.listdir(d):
        if fname.endswith("_nodelabel.txt"):
            nodelabels.append(os.path.join(d, fname))
        if "addr2funcname" in fname:
            addrfiles.append(os.path.join(d, fname))
    return sorted(nodelabels), sorted(addrfiles)

def build_func_list_from_dir(d):
    nodelabels, addrfiles = find_files_in_dir(d)
    if not nodelabels:
        raise FileNotFoundError(f"No *_nodelabel.txt found in {d}")
    # prefer the first addr file if multiple
    addrmap = {}
    if addrfiles:
        addrmap = parse_addr2funcname(addrfiles[0])
    # aggregate across multiple nodelabels if any
    agg = {}
    for nl in nodelabels:
        parsed = parse_nodelabel(nl)
        for fun, cnt in parsed.items():
            agg[fun] = agg.get(fun, 0) + cnt
    result = []
    for fun, cnt in sorted(agg.items(), key=lambda x: -x[1]):
        dec = fun_hex_to_decimal(fun)
        readable = addrmap.get(dec, "") if dec is not None else ""
        result.append({
            "fun_label": fun,
            "hex_addr": format(dec, '08x') if dec is not None else "",
            "dec_addr": dec if dec is not None else "",
            "node_count": cnt,
            "readable": readable
        })
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", "-d", required=True, help="目录，包含 *_nodelabel.txt 和 *_addr2funcname.txt")
    p.add_argument("--other_dir", "-o", help="可选，第二个目录，查找在两个目录中都有节点的函数")
    p.add_argument("--min_nodes", type=int, default=1, help="只列出节点数 >= 此阈值（默认1）")
    p.add_argument("--out", default="funcs_with_nodes.csv", help="输出 CSV 文件名")
    args = p.parse_args()

    a = build_func_list_from_dir(args.dir)
    if args.other_dir:
        b = build_func_list_from_dir(args.other_dir)
        # build set of readable names if present, else use fun_label
        b_names = set()
        for entry in b:
            if entry["readable"]:
                b_names.add(entry["readable"])
            else:
                b_names.add(entry["fun_label"])
        merged = []
        for entry in a:
            key = entry["readable"] if entry["readable"] else entry["fun_label"]
            if key in b_names and entry["node_count"] >= args.min_nodes:
                merged.append(entry)
        rows = merged
    else:
        rows = [e for e in a if e["node_count"] >= args.min_nodes]

    if not rows:
        print("未找到满足条件的函数（请确认目录与文件格式）。")
        return

    # print head
    print(f"Found {len(rows)} functions with >= {args.min_nodes} nodes in {args.dir}" +
          (f" and present in {args.other_dir}" if args.other_dir else ""))
    print("{:<6} {:<12} {:<10} {:<8} {}".format("idx","hex_addr","dec_addr","nodes","readable"))
    for i,e in enumerate(rows[:500]):
        print("{:<6} {:<12} {:<10} {:<8} {}".format(i, e["hex_addr"], str(e["dec_addr"]), e["node_count"], e["readable"]))

    # write CSV
    with open(args.out, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["fun_label","hex_addr","dec_addr","node_count","readable"])
        writer.writeheader()
        for e in rows:
            writer.writerow(e)
    print(f"\nWrote results to {args.out}")

if __name__ == "__main__":
    main()
