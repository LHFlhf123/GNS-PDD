import requests
from bs4 import BeautifulSoup
import json
import re

# 1. 指令与官网章节锚点的映射表（关键修正：同类指令共享锚点）
INSTRUCTION_TO_ANCHOR = {
    # 终止指令（Terminator Instructions）
    "ret": "terminator-instructions",
    "br": "terminator-instructions",
    "switch": "terminator-instructions",
    "indirectbr": "terminator-instructions",
    "invoke": "terminator-instructions",
    "callbr": "terminator-instructions",
    "resume": "terminator-instructions",
    "catchswitch": "terminator-instructions",
    "catchret": "terminator-instructions",
    "cleanupret": "terminator-instructions",
    "unreachable": "terminator-instructions",

    # Unary Operations
    "fneg": "unary-operations",

    # 二元运算（Binary Operations）
    "add": "binary-operations",
    "fadd": "binary-operations",
    "sub": "binary-operations",
    "fsub": "binary-operations",
    "mul": "binary-operations",
    "fmul": "binary-operations",
    "udiv": "binary-operations",
    "sdiv": "binary-operations",
    "fdiv": "binary-operations",
    "urem": "binary-operations",
    "srem": "binary-operations",
    "frem": "binary-operations",

    # 移位指令（Bitwise Operations）
    "shl": "bitwise-operations",
    "lshr": "bitwise-operations",
    "ashr": "bitwise-operations",
    "and": "bitwise-operations",
    "or": "bitwise-operations",
    "xor": "bitwise-operations",

    # 向量指令（Vector Operations）
    "extractelement": "vector-operations",
    "insertelement": "vector-operations",
    "shufflevector": "vector-operations",

    # 聚合类型指令（Aggregate Operations）
    "extractvalue": "aggregate-operations",
    "insertvalue": "aggregate-operations",

    # 内存指令（Memory Access and Addressing）
    "alloca": "memory-access-and-addressing",
    "load": "memory-access-and-addressing",
    "store": "memory-access-and-addressing",
    "fence": "memory-access-and-addressing",
    "cmpxchg": "memory-access-and-addressing",
    "atomicrmw": "memory-access-and-addressing",
    "getelementptr": "memory-access-and-addressing",

    # 类型转换指令（Cast Instructions）
    "trunc": "cast-instructions",
    "zext": "cast-instructions",
    "sext": "cast-instructions",
    "fptrunc": "cast-instructions",
    "fpext": "cast-instructions",
    "fptoui": "cast-instructions",
    "fptosi": "cast-instructions",
    "uitofp": "cast-instructions",
    "sitofp": "cast-instructions",
    "ptrtoint": "cast-instructions",
    "ptrtoaddr": "cast-instructions",
    "inttoptr": "cast-instructions",
    "bitcast": "cast-instructions",
    "addrspacecast": "cast-instructions",

    # 其他指令
    "icmp": "comparison-instructions",
    "fcmp": "comparison-instructions",
    "phi": "phi-node",
    "select": "select-instruction",
    "freeze": "freeze-instruction",
    "call": "call-instruction",
    "va_arg": "va-arg-instruction",
    "landingpad": "exception-handling-instructions",
    "catchpad": "exception-handling-instructions",
    "cleanuppad": "exception-handling-instructions"
}


# 2. 爬取主逻辑
def crawl_llvm_semantics():
    base_url = "https://llvm.org/docs/LangRef.html"
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    semantics_map = {}

    # 遍历每个指令，按共享锚点提取
    for instr, anchor in INSTRUCTION_TO_ANCHOR.items():
        print(f"正在处理指令: {instr}（章节锚点: {anchor}）")

        # 定位章节
        section = soup.find(id=anchor)
        if not section:
            print(f"⚠️ 章节不存在: {anchor}（指令: {instr}）")
            continue

        # 提取章节内所有文本段落
        paragraphs = section.find_all(['p', 'div', 'li'], recursive=True)
        full_text = "\n".join([p.get_text(strip=True, separator=" ") for p in paragraphs if p.get_text(strip=True)])

        # 匹配指令的Semantics（格式："指令名 ... Semantics: ..."）
        # 正则模式：指令名 + 任意字符 + Semantics: + 语义内容 + （下一个指令名或章节结束）
        pattern = re.compile(
            rf"{re.escape(instr)}\s+.*?Semantics:\s+(.*?)(?=\n\w+|\Z)",
            re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(full_text)

        if match:
            semantics = match.group(1).strip()
            semantics = re.sub(r'\s+', ' ', semantics)  # 清理多余空格
            semantics_map[instr] = semantics
            print(f"✅ 提取成功（长度: {len(semantics)}）")
        else:
            print(f"❌ 未找到语义: {instr}")

    # 保存结果
    with open("out/llvm_ir_semantics.json", "w", encoding="utf-8") as f:
        json.dump(semantics_map, f, indent=2, ensure_ascii=False)
    print("🎉 爬取完成，结果保存至 llvm_ir_semantics.json")


if __name__ == "__main__":
    crawl_llvm_semantics()