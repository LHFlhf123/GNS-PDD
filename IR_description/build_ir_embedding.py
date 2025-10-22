import json
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# --------------------------
# 1. 模型和路径配置（需修改为你的本地路径）
# --------------------------
# 本地Qwen3-Embedding模型路径（替换为你下载的模型目录）
local_model_path = "/home/hww/lhf/SigmaDiff-main/IR_description/Qwen3-Embedding-0.6B/"
# IR语义描述文件路径（确保与脚本同目录或填写绝对路径）
semantics_file = "llvm_ir_semantics.json"
# 嵌入结果保存路径
output_file = "llvm_ir_embeddings_qwen.json"


# --------------------------
# 2. Qwen模型要求的池化函数
# --------------------------
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """模型官方要求的last token池化逻辑"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


# --------------------------
# 3. 加载数据和模型
# --------------------------
# 读取IR指令语义描述
with open(semantics_file, "r", encoding="utf-8") as f:
    ir_semantics = json.load(f)

# 提取指令名和对应的语义文本（保持顺序）
instr_names = list(ir_semantics.keys())
instr_texts = [ir_semantics[name] for name in instr_names]

# 加载分词器和模型（本地路径，强制CPU避免CUDA问题）
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    padding_side="left"  # 模型要求left padding
)
model = AutoModel.from_pretrained(local_model_path).to("cpu")  # 强制用CPU
model.eval()


# --------------------------
# 4. 生成嵌入向量
# --------------------------
def generate_embeddings(texts, batch_size=8):
    """批量生成嵌入，适配长文本列表"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize（模型要求max_length=8192）
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        ).to(model.device)  # 输入到模型所在设备（CPU）

        # 模型前向传播
        with torch.no_grad():
            outputs = model(**batch_dict)

        # 池化+归一化（模型官方要求）
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2归一化

        # 转换为列表并收集
        all_embeddings.extend(embeddings.cpu().numpy().tolist())

    return all_embeddings


# 执行生成
print(f"开始生成嵌入，共{len(instr_texts)}条IR指令...")
ir_embeddings = generate_embeddings(instr_texts)

# 映射回指令名
result = {instr_names[i]: ir_embeddings[i] for i in range(len(instr_names))}

# --------------------------
# 5. 保存结果
# --------------------------
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"🎉 嵌入生成完成！已保存至 {output_file}（维度：{len(ir_embeddings[0])}）")