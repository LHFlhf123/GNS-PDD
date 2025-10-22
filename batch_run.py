import os
import subprocess
import re

# 基础路径配置
DATA_BASE = "/home/hww/lhf/SigmaDiff-main/data/binaries_test_all"
OUT_BASE = "/home/hww/lhf/SigmaDiff-main/out/it_function_valide_graphFirstNode"
PROJ_BASE = "/home/hww/lhf/SigmaDiff-main/tmp/batch1/"  # Ghidra项目名的基础路径

# 固定参数（不变化的部分）
FIXED_ARGS = [
    "python", "sigmadiff.py",
    "--with_gt",
    "--ghidra_home", "/home/hww/lhf/SigmaDiff-main/pack_dependence/ghidra_9.2.2/",
    "--func1", "main",
    "--func2", "main"
]

# 遍历数据目录，筛选coreutils的O2版本目录
for item in os.listdir(DATA_BASE):
    item_path = os.path.join(DATA_BASE, item)
    if not os.path.isdir(item_path):
        continue  # 跳过非目录项

    # 匹配 "coreutils-xxx-O2" 格式的目录，提取版本号xxx
    match = re.match(r"coreutils-(.*)-O2", item)
    if match:
        version = match.group(1)
        # 构造各动态参数的路径
        input1 = os.path.join(DATA_BASE, f"coreutils-{version}-O2")
        input2 = os.path.join(DATA_BASE, f"coreutils-{version}-O3")
        output_dir = os.path.join(OUT_BASE, f"coreutils-{version}-O2-O3")
        ghidra_proj_name = os.path.join(PROJ_BASE, f"coreutils-{version}-O2-O3")

        # 检查input2（O3目录）是否存在，避免无效执行
        if not os.path.exists(input2):
            print(f"警告：{input2} 不存在，跳过该版本。")
            continue

        # 确保输出目录存在，如果不存在则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"已创建输出目录：{output_dir}")
        else:
            print(f"输出目录已存在：{output_dir}")

        # 构造完整命令
        cmd = FIXED_ARGS + [
            "--input1", input1,
            "--input2", input2,
            "--output_dir", output_dir,
            "--ghidra_proj_name", ghidra_proj_name
        ]

        print(f"执行命令：{' '.join(cmd)}")
        # 执行命令并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"命令输出：\n{result.stdout}")
        print(f"命令错误：\n{result.stderr}")
        print("-" * 50)  # 分隔不同命令的输出
