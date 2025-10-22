# TestOwnDataUseModel.py
import os
# from data_utils import Corpus
# from data_utils_word2vec import Corpus
# from data_utils_doc2vec import Corpus
from data_utils_doc2vec_usemodel import Corpus
from MergingCorpus import *
from Doc2Vec import *
from pytorchtools import EarlyStopping

import torch
import torch.nn as nn
import pickle
import numpy as np

from dgmc.models import DGMC, RelCNN
from argparse import Namespace
import glob

model_path = 'llvm_3_7_0_vs_llvm_3_8_1'


# 读取 ground-truth 函数名映射（保持你原实现）
def build_func_maps(subject_dir):
    addr2real = {}
    p1 = os.path.join(subject_dir, 'addr2funcname.txt')
    if os.path.exists(p1):
        for line in open(p1):
            addr, real = line.strip().split(', ')
            addr2real[addr] = real

    addr2stripped = {}
    p2 = os.path.join(subject_dir, 'addr2funcname_stripped.txt')
    if os.path.exists(p2):
        for line in open(p2):
            addr, stripped = line.strip().split(', ')
            addr2stripped[addr] = stripped

    real2stripped = {}
    for addr, real in addr2real.items():
        stripped = addr2stripped.get(addr)
        if stripped:
            real2stripped[real] = stripped

    return real2stripped


def processDGMC(dir, filename1, filename2, args, func1=None, func2=None):
    import datetime
    print(dir, filename1, filename2, args)
    with_gt = args.with_gt
    each_conf = filename1 + '_vs_' + filename2
    subject_dir = dir + '/' + each_conf

    # 传过来的func1和fun2已经是剥离过后对应的名字了，这里直接使用
    func1s = func1
    func2s = func2
    if func1s is None or func2s is None:
        raise RuntimeError(f"无法把 {func1}/{func2} 映射到剥符号，请检查 addr2funcname*.txt")
    print(f"[FUNC-MODE] 映射可读→剥符号: {func1}->{func1s}, {func2}->{func2s}")

    # read jumps
    jump_pairs1 = []
    jp1 = os.path.join(subject_dir, "jumps1.txt")
    if os.path.exists(jp1):
        with open(jp1) as f:
            for L in f:
                s, t = L.strip().split()
                jump_pairs1.append((int(s), int(t)))

    jump_pairs2 = []
    jp2 = os.path.join(subject_dir, "jumps2.txt")
    if os.path.exists(jp2):
        with open(jp2) as f:
            for L in f:
                s, t = L.strip().split()
                jump_pairs2.append((int(s), int(t)))

    print(each_conf)
    time_1 = datetime.datetime.now()
    date_string = time_1.strftime('%b--%d')
    start_time = datetime.datetime.now()
    subject_path = subject_dir + '/'

    node_label_file_1 = os.path.join(subject_path, filename1 + "_nodelabel.txt")
    edge_file_1 = os.path.join(subject_path, filename1 + "_edges.txt")
    node_label_file_2 = os.path.join(subject_path, filename2 + "_nodelabel.txt")
    edge_file_2 = os.path.join(subject_path, filename2 + "_edges.txt")
    training_file = subject_path + "training_nodes.txt"
    func_matching_file = subject_path + "matched_functions.txt"

    corpus = Corpus()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    pretrained_subject = os.path.join(current_dir, 'casestudy/llvm_3_7_0_vs_llvm_3_8_1/not')
    ids_1_list, edges_1_list, ids_2_list, edges_2_list, train_y_list, source_type_list_list, dst_type_list_list, source_lineNum_list_list, dst_lineNum_list_list, func_matching_dict_list, src_func_dict_list, des_func_dict_list, source_value_dict_list, dst_value_dict_list, source_decompile_dict_list, dst_decompile_dict_list, node_mapping1_list, node_mapping2_list = corpus.get_data(node_label_file_1, edge_file_1, node_label_file_2, edge_file_2, training_file, func_matching_file, subject_dir, pretrained_subject, with_gt)

    filtered_indices = []
    for i, emb in enumerate(ids_1_list):
        try:
            n_nodes = emb.shape[0]
        except Exception:
            n_nodes = 0
        if n_nodes == 0:
            print(f"[SKIP] pair index {i} has zero nodes in graph1 (skipping).")
            continue
        if ids_2_list[i].shape[0] == 0:
            print(f"[SKIP] pair index {i} has zero nodes in graph2 (skipping).")
            continue
        filtered_indices.append(i)

    if len(filtered_indices) < len(ids_1_list):
        ids_1_list = [ids_1_list[i] for i in filtered_indices]
        ids_2_list = [ids_2_list[i] for i in filtered_indices]
        edges_1_list = [edges_1_list[i] for i in filtered_indices]
        edges_2_list = [edges_2_list[i] for i in filtered_indices]
        train_y_list = [train_y_list[i] for i in filtered_indices]
        print(f"[INFO] Removed {(len(filtered_indices) - len(ids_1_list))} empty pairs; continuing with {len(ids_1_list)} valid pairs.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 转换为 torch tensors（保持原有流程）
    for lst in (ids_1_list, ids_2_list):
        for j in range(len(lst)):
            lst[j] = torch.from_numpy(lst[j]).float().to(device)
    for lst in (edges_1_list, edges_2_list):
        for j in range(len(lst)):
            arr = torch.from_numpy(lst[j].astype(np.int64)).t().contiguous().to(device)
            lst[j] = arr
    for lst in (train_y_list,):
        for j in range(len(lst)):
            arr = torch.from_numpy(lst[j].astype(np.int64)).t().contiguous().to(device)
            lst[j] = arr

    # jump_index placeholders (we later may override per-subgraph)
    if len(jump_pairs1) > 0:
        try:
            jump_index_s = torch.tensor(jump_pairs1, dtype=torch.long).t().to(device)
        except Exception:
            jump_index_s = None
    else:
        jump_index_s = None
    if len(jump_pairs2) > 0:
        try:
            jump_index_t = torch.tensor(jump_pairs2, dtype=torch.long).t().to(device)
        except Exception:
            jump_index_t = None
    else:
        jump_index_t = None

    # load pretrained model
    f_model = open(os.path.join(current_dir, model_path + '_Trained_Model.pkl'), 'rb')
    model = pickle.load(f_model).to(device)
    fusion_dim = args.dim
    model.jump_fusion = nn.Linear(2 * fusion_dim, fusion_dim, bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    vocab_size = len(corpus.dictionary)

    # 获取真正的原始 subject_dir（用于输出文件名一致）
    if subject_dir.endswith("_Finetuned-results"):
        result_dir = subject_dir  # 兼容直接作为输出目录传入
        # subject_name = os.path.basename(subject_dir[:-len("_Finetuned-results")])
    else:
        result_dir = subject_dir + '_Finetuned'
        # subject_name = os.path.basename(subject_dir)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 创建文件，确保 evaluate.py 能找到对应的 *-match_result.txt 文件
    # match_path = os.path.join(result_dir, f"{subject_name}-match_result.txt")
    # init_path = os.path.join(result_dir, f"{subject_name}-Initial_match_result.txt")
    # create empty if not exists (safe)

    # 单函数运行模式（保持你现有行为）
    if isinstance(args, Namespace) and func1 and func2:
        match_idx = None
        for idx, (src_map, dst_map) in enumerate(zip(src_func_dict_list, des_func_dict_list)):
            if set(src_map.values()) == {func1s} and set(dst_map.values()) == {func2s}:
                match_idx = idx
        if match_idx is None:
            raise RuntimeError(f"找不到子图 '{func1s}' ↔ '{func2s}'，请检查 matched_functions.txt 与 --func1/--func2")
        indexes = [match_idx]
    else:
        indexes = list(range(len(ids_1_list)))
        indexes.reverse()

    for i in indexes:
        ids_1 = ids_1_list[i]
        edges_1 = edges_1_list[i]
        ids_2 = ids_2_list[i]
        edges_2 = edges_2_list[i]
        train_y = train_y_list[i]

        print(f"[PRE-DGMC] func_pair_index={i} src_nodes={ids_1.size(0)} tgt_nodes={ids_2.size(0)}")

        # per-subgraph jump mapping (use node_mappingX_list if available)
        if node_mapping1_list is not None:
            nm1 = node_mapping1_list[i]
            local_jumps_s = []
            for src_nid, tgt_nid in jump_pairs1:
                if src_nid in nm1 and tgt_nid in nm1:
                    local_jumps_s.append((nm1[src_nid], nm1[tgt_nid]))
            if local_jumps_s:
                jump_index_s = torch.tensor(local_jumps_s, dtype=torch.long).t().to(device)
            else:
                jump_index_s = None
        else:
            jump_index_s = None

        if node_mapping2_list is not None:
            nm2 = node_mapping2_list[i]
            local_jumps_t = []
            for src_nid, tgt_nid in jump_pairs2:
                if src_nid in nm2 and tgt_nid in nm2:
                    local_jumps_t.append((nm2[src_nid], nm2[tgt_nid]))
            if local_jumps_t:
                jump_index_t = torch.tensor(local_jumps_t, dtype=torch.long).t().to(device)
            else:
                jump_index_t = None
        else:
            jump_index_t = None

        source_type_list = source_type_list_list[i]
        dst_type_list = dst_type_list_list[i]
        source_lineNum_list = source_lineNum_list_list[i]
        dst_lineNum_list = dst_lineNum_list_list[i]
        func_matching_dict = func_matching_dict_list[i]
        src_func_dict = src_func_dict_list[i]
        des_func_dict = des_func_dict_list[i]
        source_value_dict = source_value_dict_list[i]
        dst_value_dict = dst_value_dict_list[i]
        source_decompile_dict = source_decompile_dict_list[i]
        dst_decompile_dict = dst_decompile_dict_list[i]

        if node_mapping1_list is None:
            node_mapping1 = {ids: ids for ids in range(len(ids_1))}
            node_mapping2 = {ids: ids for ids in range(len(ids_2))}
        else:
            node_map1 = node_mapping1_list[i]
            node_map2 = node_mapping2_list[i]
            try:
                node_mapping1 = {node_map1[key]: key for key in node_map1.keys()}
                node_mapping2 = {node_map2[key]: key for key in node_map2.keys()}
            except Exception:
                node_mapping1 = {idx: idx for idx in range(len(ids_1))}
                node_mapping2 = {idx: idx for idx in range(len(ids_2))}

        def train():
            model.train()
            optimizer.zero_grad()
            _, S_L = model(ids_1, edge_index_1, None, None, ids_2,
                           edge_index_2, None, None, train_y, jump_index_s=jump_index_s, jump_index_t=jump_index_t)
            loss = model.loss(S_L, train_y, source_type_list, dst_type_list, source_value_dict, dst_value_dict, source_decompile_dict, dst_decompile_dict, source_lineNum_list, dst_lineNum_list)
            loss.backward()
            optimizer.step()
            if hasattr(model, 'jump_fusion') and hasattr(model.jump_fusion, 'weight'):
                print(f"[CHECK][train step] jump_fusion.weight norm:", model.jump_fusion.weight.norm().item())
            return loss

        @torch.no_grad()
        def test(final=False):
            model.eval()
            _, S_L = model(ids_1, edge_index_1, None, None, ids_2,
                           edge_index_2, None, None, None, jump_index_s=jump_index_s, jump_index_t=jump_index_t)
            accuracy = model.accdiff(S_L, source_lineNum_list, dst_lineNum_list, func_matching_dict, src_func_dict, des_func_dict, source_type_list, dst_type_list, None, source_value_dict, dst_value_dict, subject_path, result_dir, node_mapping1, node_mapping2, final, with_gt)
            return accuracy

        # DEBUG prints (保持)
        print(ids_1.size())
        print(ids_2.size())
        all_nodes = float(ids_1.size()[0])
        training_nodes = float(train_y.size()[0])
        print(all_nodes)
        print(training_nodes)
        print(float(training_nodes) / all_nodes)
        print(edges_1.size())
        print(ids_2.size())
        print(edges_2.size())

        # ensure edges orientation; these are local tensors shaped [2, E] as earlier converted
        if edges_1.dim() == 2 and edges_1.size(0) == 2 and edges_1.size(1) != 2:
            edges_1 = edges_1.t()
            print("  transposed edges_1 →", edges_1.size())
        if edges_2.dim() == 2 and edges_2.size(0) == 2 and edges_2.size(1) != 2:
            edges_2 = edges_2.t()
            print("  transposed edges_2 →", edges_2.size())

        # Convert to expected names by DGMC model
        edge_index_1 = edges_1.t()
        edge_index_2 = edges_2.t()
        train_y = train_y.t()
        print("train_y.size() =", train_y.size())

        # NOTE: preserve original tensors from lists (the model call expects them in that form)
        ids_1 = ids_1_list[i]
        edges_index_1 = edges_1_list[i]
        ids_2 = ids_2_list[i]
        edges_index_2 = edges_2_list[i]
        train_y = train_y_list[i]

        patience = 30
        early_stopping = EarlyStopping(patience, verbose=True)

        print('Optimize initial feature matching...')
        model.num_steps = 0
        model.detach = False
        for epoch in range(1, 161):
            if epoch == 140:
                print('Refine correspondence matrix...')
                model.num_steps = args.num_steps
                model.detach = True

            loss = train()
            early_stopping(loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                accuracy = test(final=True)
                break

            if epoch == 160:
                accuracy = test(final=True)
                f_model = open('Trained_Model.pkl', 'wb')
                pickle.dump(model, f_model, protocol=4)

            if epoch % 10 == 0 or epoch > 160:
                accuracy = test()
                print(subject_path + ':' + (f'{epoch:03d}: Loss: {loss:.4f}'))

        # cleanup per-iteration allocations
        del ids_1, edges_1, ids_2, edges_2
        del early_stopping
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # NOTE: this main is only example — your sigmadiff.py probably calls processDGMC()
    processDGMC("/mnt/sata/lian/github/SigmaDiff/out", "diffutils-2.8-O0_cmpstripped", "diffutils-2.8-O3_cmpstripped", None)
