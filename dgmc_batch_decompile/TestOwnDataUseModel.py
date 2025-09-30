import os
# from data_utils import Corpus
# from data_utils_word2vec import Corpus
# from data_utils_doc2vec import Corpus
from data_utils_doc2vec_usemodel import Corpus
from MergingCorpus import *
from Doc2Vec import *
from pytorchtools import EarlyStopping

import torch
import os
import torch.nn as nn
import pickle
import numpy as np

from dgmc.models import DGMC, RelCNN
from argparse import Namespace

model_path = 'llvm_3_7_0_vs_llvm_3_8_1'


# 读取 ground-truth 函数名映射
def build_func_maps(subject_dir):
    # 1) 从 addr2funcname.txt 读出: addr → real_name
    addr2real = {}
    p1 = os.path.join(subject_dir, 'addr2funcname.txt')
    if os.path.exists(p1):
        for line in open(p1):
            addr, real = line.strip().split(', ')
            addr2real[addr] = real

    # 2) 从 addr2funcname_stripped.txt 读出: addr → stripped_name
    addr2stripped = {}
    p2 = os.path.join(subject_dir, 'addr2funcname_stripped.txt')
    if os.path.exists(p2):
        for line in open(p2):
            addr, stripped = line.strip().split(', ')
            addr2stripped[addr] = stripped

    # 3) 反转成 real_name → stripped_name
    real2stripped = {}
    for addr, real in addr2real.items():
        stripped = addr2stripped.get(addr)
        if stripped:
            real2stripped[real] = stripped

    return real2stripped






def processDGMC(dir, filename1, filename2, args, func1=None, func2=None):
    print(dir, filename1, filename2, args)
    with_gt = args.with_gt
    each_conf =  filename1 + '_vs_' + filename2
    subject_dir=dir+'/'+each_conf


    # ===== 新增：把 main → FUN_… 转换 =====
    # stripped_map1 = load_stripped_map(subject_dir)
    # if func1 in stripped_map1:
    #     func1_stripped = stripped_map1[func1]
    # else:
    #     func1_stripped = func1
    # stripped_map2 = load_stripped_map(subject_dir)
    # if func2 in stripped_map2:
    #     func2_stripped = stripped_map2[func2]
    # else:
    #     func2_stripped = func2
    # ——— 映射真实名 → 剥符号名 ———
    # real2stripped = build_func_maps(subject_dir)
    # print("[DEBUG] real2stripped mapping:", real2stripped)
    # func1s = real2stripped.get(func1)
    # func2s = real2stripped.get(func2)

    # 传过来的func1和fun2已经是剥离过后对应的名字了，这里直接使用
    func1s = func1
    func2s = func2
    if func1s is None or func2s is None:
        raise RuntimeError(
          f"无法把 {func1}/{func2} 映射到剥符号，请检查 addr2funcname*.txt"
        )
    print(f"[FUNC-MODE] 映射可读→剥符号: {func1}->{func1s}, {func2}->{func2s}")


    # 读入 jumps1.txt/jumps2.txt   *加
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

    # if(not os.path.exists(each_conf+'_Trained_Model.pkl')):
    #     continue

    time_1=datetime.datetime.now()
    date_string=time_1.strftime('%b--%d')

    # result_file=open(subject_dir+'-'+date_string+'_UseModel_Results.csv','w')
    # result_file=open(subject_dir.replace('/','--')+'-'+date_string+'_UseModel_FurtherTraining_Results.csv','w')
    # result_file.write('System,TrainingNodePercent,FinalAccuracy,Time,EarlyStop\n')
    # result_file.flush()

    start_time=datetime.datetime.now()

    subject_path=subject_dir+'/'
    
    ###############################################################
    node_label_file_1= os.path.join(subject_path, filename1 + "_nodelabel.txt")
    edge_file_1= os.path.join(subject_path, filename1 + "_edges.txt")
    node_label_file_2= os.path.join(subject_path, filename2 + "_nodelabel.txt")
    edge_file_2= os.path.join(subject_path, filename2 + "_edges.txt")
    training_file=subject_path+"training_nodes.txt"
    func_matching_file=subject_path+"matched_functions.txt"
    ###############################################################

    # use pretrained doc2vec
    corpus = Corpus()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    pretrained_subject = os.path.join(current_dir, 'casestudy/llvm_3_7_0_vs_llvm_3_8_1/not')
    ids_1_list,edges_1_list,ids_2_list,edges_2_list,train_y_list,source_type_list_list,dst_type_list_list,source_lineNum_list_list,dst_lineNum_list_list,func_matching_dict_list,src_func_dict_list,des_func_dict_list,source_value_dict_list,dst_value_dict_list,source_decompile_dict_list,dst_decompile_dict_list, node_mapping1_list, node_mapping2_list = corpus.get_data(node_label_file_1,edge_file_1,node_label_file_2,edge_file_2,training_file,func_matching_file,subject_dir,pretrained_subject, with_gt)

    filtered_indices = []
    for i, emb in enumerate(ids_1_list):  # ids_1_list 对应 embedding_1_list 名称，按你代码实际变量名调整
        # emb can be numpy array or torch tensor; check shape[0]
        try:
            n_nodes = emb.shape[0]
        except Exception:
            n_nodes = 0
        if n_nodes == 0:
            print(f"[SKIP] pair index {i} has zero nodes in graph1 (skipping).")
            continue
        # also check graph2
        if ids_2_list[i].shape[0] == 0:
            print(f"[SKIP] pair index {i} has zero nodes in graph2 (skipping).")
            continue
        filtered_indices.append(i)

    # regen all lists to only include valid indices
    if len(filtered_indices) < len(ids_1_list):
        ids_1_list = [ids_1_list[i] for i in filtered_indices]
        ids_2_list = [ids_2_list[i] for i in filtered_indices]
        edges_1_list = [edges_1_list[i] for i in filtered_indices]
        edges_2_list = [edges_2_list[i] for i in filtered_indices]
        train_y_list = [train_y_list[i] for i in filtered_indices]
        # ... and any other parallel lists returned by get_data
        print(
            f"[INFO] Removed {(len(filtered_indices) - len(ids_1_list))} empty pairs; continuing with {len(ids_1_list)} valid pairs.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # —— 将 get_data 返回的 NumPy arrays list 全部转成 PyTorch tensors ——
    for lst in (ids_1_list, ids_2_list):
        # 每个元素 shape [N, D]
        for j in range(len(lst)):
            lst[j] = torch.from_numpy(lst[j]).float().to(device)
    for lst in (edges_1_list, edges_2_list):
        # 每个元素 shape [E, 2] → [2, E]
        for j in range(len(lst)):
            arr = torch.from_numpy(lst[j].astype(np.int64)).t().contiguous().to(device)
            lst[j] = arr
    for lst in (train_y_list,):
        # 每个元素 shape [T, 2] → [2, T]
        for j in range(len(lst)):
            arr = torch.from_numpy(lst[j].astype(np.int64)).t().contiguous().to(device)
            lst[j] = arr





    # *加
    jump_index_s = torch.tensor(jump_pairs1, dtype=torch.long).t().to(device)  # shape [2, E_s]
    jump_index_t = torch.tensor(jump_pairs2, dtype=torch.long).t().to(device)


    # # —— 先加载预训练模型，并把它带入循环 ——
    f_model = open(os.path.join(current_dir, model_path + '_Trained_Model.pkl'), 'rb')
    model = pickle.load(f_model).to(device)
    # # 手工挂载 jump_fusion（确保 in_channels 和 device 都对得上）
    fusion_dim = args.dim  # 你 run sigmadiff 时传给 --dim 的值，通常是 128
    model.jump_fusion = nn.Linear(2 * fusion_dim, fusion_dim, bias=True).to(device)
    # # 然后再创建优化器，保证它会把 jump_fusion 的参数也加进去
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # # 打印验证模型是否增加了线性层
    # print(">> Has jump_fusion?", hasattr(model, "jump_fusion"), model.jump_fusion)
    #
    # # —— 打印 jump_fusion 初始权重范数 ——
    # if hasattr(model, 'jump_fusion') and hasattr(model.jump_fusion, 'weight'):
    #     print("[CHECK] Initial jump_fusion.weight norm:", model.jump_fusion.weight.norm().item())

    vocab_size = len(corpus.dictionary)
    subject_name=subject_path.strip('/').replace('/','-')
    result_dir=subject_dir+'_Finetuned-results'

    if(not os.path.exists(result_dir)):
        os.mkdir(result_dir)

    match_file=open(result_dir+'/'+subject_name+'-match_result.txt','w')
    before_filtering_match=open(result_dir+'/'+subject_name+'-Initial_match_result.txt','w')
    match_file.close()
    before_filtering_match.close()


    # indexes = list(range(len(ids_1_list)))
    # indexes.reverse()

    # 改为单函数运行
    # —— 单函数模式：先寻找 func1/func2 在 ids_?_list 中对应的子图索引 ——
    # if isinstance(args, Namespace) and func1 and func2:
    #     match_idx = None
    #     for idx, (src_map, dst_map) in enumerate(zip(src_func_dict_list, des_func_dict_list)):
    #         # src_map: dict local_node_id -> 函数名，理想情况 whole map 全部都是 func1
    #         if set(src_map.values()) == {func1_stripped} and set(dst_map.values()) == {func2_stripped}:
    #             match_idx = idx
    #             break
    #     if match_idx is None:
    #         raise RuntimeError(f"找不到与函数 '{func1}'(→{func1_stripped}) ↔ "
    #                            f"'{func2}'(→{func2_stripped}) 对应的子图。"
    #                            "请确认 matched_functions.txt 和 --func1/--func2 是否一致。")
    #     indexes = [match_idx]
    # else:
    #     # 全程序模式
    #     indexes = list(range(len(ids_1_list)))
    #     indexes.reverse()

    if isinstance(args, Namespace) and func1 and func2:
        # ——— 单函数：匹配 剥符号 名称 func1_stripped vs func2_stripped ———
        match_idx = None
        for idx, (src_map, dst_map) in enumerate(zip(src_func_dict_list, des_func_dict_list)):
            if set(src_map.values()) == {func1s} and set(dst_map.values()) == {func2s}:
                match_idx = idx
                # break
        if match_idx is None:
            raise RuntimeError(
                f"找不到子图 '{func1s}' ↔ '{func2s}'，"
                "请检查 matched_functions.txt 与 --func1/--func2"
            )
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

        # --------- Diagnostic print A: before entering DGMC (node counts) ----------
        # 打印A进图匹配前的节点数
        # def _node_count(x):
        #     # supports torch.Tensor, numpy.ndarray, list, or Python object with shape/len
        #     try:
        #         import torch
        #         if isinstance(x, torch.Tensor):
        #             return int(x.size(0))
        #     except Exception:
        #         pass
        #     try:
        #         import numpy as _np
        #         if isinstance(x, _np.ndarray):
        #             return int(x.shape[0])
        #     except Exception:
        #         pass
        #     try:
        #         return int(len(x))
        #     except Exception:
        #         return -1
        #
        # src_nodes = _node_count(ids_1)
        # tgt_nodes = _node_count(ids_2)
        # print(f"[PRE-DGMC] func_pair_index={i} src_nodes={src_nodes} tgt_nodes={tgt_nodes}")
        print(f"[PRE-DGMC] func_pair_index={i} src_nodes={ids_1.size(0)} tgt_nodes={ids_2.size(0)}")




        # 在 for i in indexes: 循环里，紧接 ids_1, ids_2, train_y 取得之后  *加
        # —— 根据 node_mapping*_list 来为本子图构造 jump_index_s/jump_index_t ——
        if node_mapping1_list is not None:
            nm1 = node_mapping1_list[i]   # dict: global_node_id -> local_idx
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


        # —— 外部特征级联 —— 在最原始 ids_1/ids_2 张量上做拼接+投影 ——

        # for src_idx, tgt_idx in jump_pairs1:
        #     if 0 <= tgt_idx < ids_1.size(0) and 0 <= src_idx < ids_1.size(0):
        #         cat = torch.cat([ids_1[tgt_idx], ids_1[src_idx]], dim=-1)  # [2D]
        #         with torch.no_grad():
        #             fused = model.jump_fusion(cat.to(device)).cpu()  # 投影回 D，再搬回 CPU
        #         ids_1[tgt_idx] = fused
        #
        # for src_idx, tgt_idx in jump_pairs2:
        #     if 0 <= tgt_idx < ids_2.size(0) and 0 <= src_idx < ids_2.size(0):
        #         cat = torch.cat([ids_2[tgt_idx], ids_2[src_idx]], dim=-1)
        #         with torch.no_grad():
        #             fused = model.jump_fusion(cat.to(device)).cpu()
        #         ids_2[tgt_idx] = fused
        #
        # print(f"[DGMC·FUSE] #jumps_s={len(jump_pairs1)}, #jumps_t={len(jump_pairs2)}")





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
            node_mapping1 = {ids:ids for ids in range(len(ids_1))}
            node_mapping2 = {ids:ids for ids in range(len(ids_2))}
        else:
            node_map1 = node_mapping1_list[i]
            node_map2 = node_mapping2_list[i]
            node_mapping1 = {node_map1[key]:key for key in node_map1.keys()}
            node_mapping2 = {node_map2[key]:key for key in node_map2.keys()}
        def train():
            model.train()
            optimizer.zero_grad()

            _, S_L = model(ids_1, edge_index_1, None, None, ids_2,
                        edge_index_2, None, None, train_y ,jump_index_s=jump_index_s,jump_index_t=jump_index_t)

            loss = model.loss(S_L, train_y, source_type_list, dst_type_list, source_value_dict, dst_value_dict, source_decompile_dict, dst_decompile_dict, source_lineNum_list, dst_lineNum_list)

            loss.backward()
            optimizer.step()
            # —— 打印 jump_fusion 每步更新后的权重范数 ——
            if hasattr(model, 'jump_fusion') and hasattr(model.jump_fusion, 'weight'):
                print(f"[CHECK][train step] jump_fusion.weight norm:", model.jump_fusion.weight.norm().item())
            return loss

        @torch.no_grad()
        def test(final=False):
            model.eval()
            _, S_L = model(ids_1, edge_index_1, None, None, ids_2,
                        edge_index_2, None, None, None, jump_index_s=jump_index_s,jump_index_t=jump_index_t)

            accuracy = model.accdiff(S_L, source_lineNum_list,dst_lineNum_list,func_matching_dict,src_func_dict,des_func_dict,source_type_list,dst_type_list,None,source_value_dict,dst_value_dict,subject_path,result_dir, node_mapping1, node_mapping2, final, with_gt)
            return accuracy

        print(ids_1.size())
        print(ids_2.size())

        all_nodes=float(ids_1.size()[0])
        training_nodes=float(train_y.size()[0])
        print(all_nodes)
        print(training_nodes)
        print(float(training_nodes)/all_nodes)

        print(edges_1.size())
        print(ids_2.size())
        print(edges_2.size())

        # ——— 保证 edges_1 是 [num_edges, 2] 而不是 [2, num_edges] ———
        if edges_1.dim() == 2 and edges_1.size(0) == 2 and edges_1.size(1) != 2:
            edges_1 = edges_1.t()
            print("  transposed edges_1 →", edges_1.size())

        if edges_2.dim() == 2 and edges_2.size(0) == 2 and edges_2.size(1) != 2:
            edges_2 = edges_2.t()
            print("  transposed edges_2 →", edges_2.size())


        edge_index_1=edges_1.t()
        print("edge_index_1.size() =", edge_index_1.size())

        edge_index_2=edges_2.t()
        train_y=train_y.t()
        print("train_y.size() =", train_y.size())

        # ids_1=ids_1.to(device)
        # # ids_1 = torch.from_numpy(ids_1).to(device)
        # edge_index_1=edge_index_1.to(device)
        # ids_2=ids_2.to(device)
        # # ids_2 = torch.from_numpy(ids_2).to(device)
        # edge_index_2=edge_index_2.to(device)
        # train_y=train_y.to(device)

        # 取出已经转好的 tensors
        ids_1   = ids_1_list[i]
        edges_index_1 = edges_1_list[i]
        ids_2   = ids_2_list[i]
        edges_index_2 = edges_2_list[i]
        train_y = train_y_list[i]



        # f_model=open(os.path.join(current_dir, model_path+'_Trained_Model.pkl'),'rb')
        # model = pickle.load(f_model)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # # —— 打印 jump_fusion 初始权重范数 ——
        # if hasattr(model, 'jump_fusion') and hasattr(model.jump_fusion, 'weight'):
        #     print("[CHECK] Initial jump_fusion.weight norm:", model.jump_fusion.weight.norm().item())

        patience = 30	# 当验证集损失在连续30次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
        early_stopping = EarlyStopping(patience, verbose=True)

        result_dir=subject_dir+'_Finetuned'

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
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("Early stopping")
                # 结束模型训练
                # end_time=datetime.datetime.now()
                accuracy=test(final=True)
                # result_file.write(each_conf+','+str(float(training_nodes)/all_nodes)+','+str(accuracy)+','+str((end_time-start_time).total_seconds())+',Yes\n')
                # result_file.flush()
                break

            if epoch == 160:
                # end_time=datetime.datetime.now()

                accuracy=test(final=True)
                # result_file.write(each_conf+','+str(float(training_nodes)/all_nodes)+','+str(accuracy)+','+str((end_time-start_time).total_seconds())+',No\n')
                # result_file.flush()

                f_model=open('Trained_Model.pkl','wb')
                pickle.dump(model, f_model, protocol = 4)

            if epoch % 10 == 0 or epoch > 160:
                accuracy=test()
                print(subject_path+':'+(f'{epoch:03d}: Loss: {loss:.4f}'))
        
        # del model
        del ids_1,edges_1,ids_2,edges_2
        del early_stopping
        torch.cuda.empty_cache()

if __name__ == "__main__":
    processDGMC("/mnt/sata/lian/github/SigmaDiff/out", "diffutils-2.8-O0_cmpstripped", "diffutils-2.8-O3_cmpstripped", None)