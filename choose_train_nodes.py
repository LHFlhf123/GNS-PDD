from ast import dump
from operator import add, truediv
import os
import subprocess
import errno
import shutil
from collections import defaultdict


# 只处理两个函数

def filter_to_function(nodetxt_in, nodetxt_out, func_name):
    """只保留 nodelabel.txt 中 #func_name 那一段。"""
    with open(nodetxt_in) as fi, open(nodetxt_out, 'w') as fo:
        keep = False
        for line in fi:
            if line.startswith("#"):
                keep = (line[1:].strip() == func_name)
                if keep: fo.write(line)
            elif keep:
                fo.write(line)







# 新增处理跳转指令的函数

# 1.1 读取 nodelabel.txt，构建 node_id -> 特征 list 映射
def load_node_features_and_mapping(node_label_file):
    node_features = {}
    id2idx = {}
    idx = 0
    with open(node_label_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"): continue
            parts = line.split("|&|")
            nid = parts[0]  # e.g. "136466"
            node_features[nid] = parts
            id2idx[nid] = idx  # 把它在这一函数／图里的顺序记下来(ids本来就是整数不要转成字符串)
            idx += 1
    return node_features, id2idx

# 1.2 在原 edges.txt 基础上，向跳转目标添加一条 type=4 JUMP 边
def augment_edges_with_jumps(edge_in, edge_out, node_features,id2idx):
    # 先把原有所有边读进来
    with open(edge_in) as f: base_edges = [l.rstrip() for l in f]
    print(f"[JUMP-AUG] loading {len(base_edges)} base edges from {edge_in}")

    jump_pair = []    #  用来存真正要返回的 (src, tgt)
    jump_edges = []   #  用来写到 edge_out 的 "src, tgt, 4" 字符串
    for nid, parts in node_features.items():
        tag = parts[3]  # e.g. "CBRANCH@@..." or "BRANCH@@dest_id"
        # 只要 opcode 是 CBRANCH 或 BRANCH，就认为它有跳转目标存 parts[4]
        if tag.startswith("CBRANCH@@") or tag.startswith("BRANCH@@"):
            # tgt = parts[4]
            # if tgt in node_features:
            #     jump_edges.append(f"{nid}, {tgt}, 4")  # type=4  extra/JUMP

            # 拆出 target,   从 CBRANCH@@<target>##... 中拆出 <target>
            tgt_raw = tag.split("@@")[1].split("##",1)[0]
            # 看这个 target_raw 是不是我们 id2idx 里的一个局部编号
            # 现在 tgt_raw 是原始节点 ID，
            # 我们要看它是否在 id2idx 的 keys 中
            # if tgt_raw.isdigit() and tgt_raw in id2idx.values():
            if tgt_raw in id2idx:
                src_idx = id2idx[nid]
                tgt_idx = id2idx[tgt_raw]
                jump_pair.append((src_idx, tgt_idx))
                jump_edges.append(f"{src_idx}, {tgt_idx}, 4")
    print(f"[JUMP-AUG] found {len(jump_edges)} jump edges")

    # 写回所有边
    with open(edge_out, "w") as f:
        for e in base_edges:    f.write(e + "\n")
        for je in jump_edges:   f.write(je + "\n")

    # 返回真正的跳转对  *加
    return jump_pair

    # print(f"[JUMP-AUG] wrote {len(base_edges) + len(jump_edges)} edges to {edge_out}")

    # return jump_edges


def load_ast_nodes(file_path):
    node_features = dict()
    with open(file_path, "r") as f:
        lines = f.readlines()
    funcname = None
    func_features = None
    func_nodenames = None
    func_nodes = {}
    node_features_full = {}
    node_names_full = {}
    new_lines = []
    string_libcall_id = {}
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            funcname = line[1:]
            func_features = defaultdict(set)
            func_nodenames = defaultdict(set)
        else:
            features = line.split("|&|")
            if funcname not in func_nodes:
                func_nodes[funcname] = [features[0]]
            if features[1] == 'RETURN':
                func_nodes[funcname].append(features[0])
            elif features[1] == 'ARG1':
                func_nodes[funcname].append(features[0])
            elif features[1] == 'ARG2':
                func_nodes[funcname].append(features[0])
            elif features[1] == 'ARG3':
                func_nodes[funcname].append(features[0])
            elif features[1] == 'ARG4':
                func_nodes[funcname].append(features[0])
            elif features[1] == 'ARG5':
                func_nodes[funcname].append(features[0])
            elif features[1] == 'ARG6':
                func_nodes[funcname].append(features[0])
            if len(features) < 4:
                print(line)
            if features[3] != "null":
                if "@@" in features[3]:
                    seg = features[3].split("@@")
                    if seg[0] in ['BOOL_AND', 'BOOL_OR', 'FLOAT_EQUAL', 'FLOAT_NOTEQUAL', 'FLOAT_LESS', 'FLOAT_LESSEQUAL', 'INT_EQUAL', 'INT_NOTEQUAL', 'INT_SLESS', 'INT_SLESSEQUAL', 'INT_LESS', 'INT_LESSEQUAL']:
                        features[3] = "CMP@@"+seg[1]
                func_features[features[3]].add(features[0])
            func_nodenames[features[1]].add(features[0])
            node_features_full[features[0]] = features[1:]
            if features[3] == "LIBCALL" or features[3] == "STR":
                string_libcall_id[features[1]] = features[0]

        node_features[funcname] = func_features
        node_names_full[funcname] = func_nodenames

    return node_features, func_nodes, node_features_full, node_names_full, string_libcall_id


def add_attr_ast_nodes(file_path, addrmap):
    node_features = dict()
    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        for line in lines:
            if line.startswith("#"):
                f.write(line)
            else:
                line = line.strip()
                features = line.split("|&|")
                if features[3].find("@@") != -1:
                    seg = features[3].split("@@")
                    if seg[0] in ['BOOL_AND', 'BOOL_OR', 'FLOAT_EQUAL', 'FLOAT_NOTEQUAL', 'FLOAT_LESS', 'FLOAT_LESSEQUAL', 'INT_EQUAL', 'INT_NOTEQUAL', 'INT_SLESS', 'INT_SLESSEQUAL', 'INT_LESS', 'INT_LESSEQUAL']:
                        features[3] = "CMP"
                    else:
                        features[3] = features[3].split("@@")[0]
                if features[-1] != "null":
                    addr_set = [a for a in features[-1].split("##") if a !=""]
                    src_line_set = set()
                    for addr in addr_set:
                        addr = addr.lstrip("0")
                        if not addrmap is None and addr in addrmap:
                            if addrmap[addr].startswith("??") or addrmap[addr].endswith("?"):
                                continue
                            if addrmap[addr].split("/")[-2] == '.':
                                src_line = "/".join([addrmap[addr].split("/")[-3],addrmap[addr].split("/")[-1]])
                            else:
                                src_line = "/".join(addrmap[addr].split("/")[-2:])
                            if src_line.endswith(")"):
                                ind = src_line.rfind("(")
                                src_line = src_line[:ind-1]
                            src_line_set.add(src_line)
                        else:
                            continue
                    if len(src_line_set) == 0:
                        features.append("null")
                        f.write("|&|".join(features) + '\n')
                    else:
                        features.append("##".join(src_line_set))
                        f.write("|&|".join(features) + '\n')
                else:
                    f.write(line + '|&|null\n')


def get_base_address(bin_path):
    cmd = "readelf -l " + bin_path + " | grep LOAD"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    base = str(out).split()[3]
    return int(base, 16)


def get_source_lines(bin_file, outputfile, node_features_full, ghidra_base):
    offset = get_base_address(bin_file)
    real_offset = offset - ghidra_base
    addresses_query = []
    for nodeid in node_features_full:
        addresses = node_features_full[nodeid][-1].split("##")
        for addr in addresses:
            if len(addr) > 2 and addr != "null":
                addr = int(addr, 16) + real_offset
                addresses_query.append(hex(addr)[2:])

    addr2file = outputDebugInfo(bin_file, outputfile, addresses_query, real_offset)
    return addr2file


def outputDebugInfo(bin_file, output_file, addresses_query, offset):
    f = open(output_file, 'w')
    f.close()

    for i in range((len(addresses_query) + 500 - 1) // 500):
        cmd = 'addr2line -e {} -a {}'.format(bin_file, " ".join(addresses_query[i * 500:(i + 1) * 500]))
        if not os.path.exists(os.path.dirname(output_file)):
            try:
                os.makedirs(os.path.dirname(output_file))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(output_file, 'a') as debugInfo:
            p = subprocess.Popen(cmd, shell=True, stdout=debugInfo, close_fds=True)
            p.wait()

    with open(output_file, "r") as f:
        dumpinfo = f.readlines()
    addr2file = {}
    for i, addr in enumerate(addresses_query):
        file_info = dumpinfo[(i+1)*2-1]
        addr = int(addr, 16) - offset  # need to substract the offset added when query addr2line
        addr2file[hex(addr)[2:]] = file_info.strip()

    return addr2file


def filter_different_lines(map1, map2):
    keys1 = [k for k in map1.keys() if map1[k] in set(map1.values())-set(map2.values())]
    keys2 = [k for k in map2.keys() if map2[k] in set(map2.values())-set(map1.values())]
    [map1.pop(k) for k in keys1]
    [map2.pop(k) for k in keys2]


def load_edges(file):
    with open(file, "r") as f:
        lines = f.readlines()
    graph = defaultdict(set)
    graph_reverse = defaultdict(set)
    for line in lines:
        line = line.strip()
        src, des, type = line.split(", ")
        if type == '1':
            graph[src].add(des)
            graph_reverse[des].add(src)

    return graph, graph_reverse


def find_unchanged_functions(corpus_file1, corpus_file2):
    func_corpus1 = dict()
    func_corpus2 = dict()
    func_lines1 = dict()
    func_lines2 = dict()
    with open(corpus_file1, "r") as f:
        lines = f.readlines()
    funcname = None
    newlines = ''
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            funcname = line[1:]
            func_corpus1[funcname] = []
            func_lines1[funcname] = ""
        else:
            nodeid = line.split('##')[0]
            strippedline = line[len(nodeid)+2:]
            func_corpus1[funcname].append(nodeid)
            func_lines1[funcname] += strippedline + ' '
            newlines += strippedline + '\n'

    with open(corpus_file1, "w") as f:
        f.write(newlines)


    with open(corpus_file2, "r") as f:
        lines = f.readlines()
    funcname = None
    newlines = ''
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            funcname = line[1:]
            func_corpus2[funcname] = []
            func_lines2[funcname] = ""
        else:
            nodeid = line.split('##')[0]
            strippedline = line[len(nodeid)+2:]
            func_corpus2[funcname].append(nodeid)
            func_lines2[funcname] += strippedline + ' '
            newlines += strippedline + '\n'

    with open(corpus_file2, "w") as f:
        f.write(newlines)
    return func_corpus1, func_lines1, func_corpus2, func_lines2

def add_training_nodes(file1, node_features_full1, file2, node_features_full2, matched_pair):
    g1, gr1 = load_edges(file1)
    g2, gr2 = load_edges(file2)
    added_pairs = []
    for n1, n2 in matched_pair:
        parent1 = gr1[n1]
        parent2 = gr2[n2]
        if len(parent1) == 1 and len(parent2) == 1:
            parent1 = list(parent1)[0]
            parent2 = list(parent2)[0]
            if node_features_full1[parent1][1] == "ClangStatement" and node_features_full2[parent2][1] == "ClangStatement":
                added_pairs.append((parent1, parent2))

    return added_pairs


def merge(dict1, dict2):
    for key in dict2.keys():
        dict1[key].update(dict2[key])


def select_training_node(node_file1, node_file2, matched_functions, training_node_path, node_file_new1, node_file_new2, bin_path1, bin_path2, debug_info1, debug_info2, corpus_file_new1, corpus_file_new2, base1, base2, with_gt):
    node_features1, func_nodes1, node_features_full1, node_names_full1, string_libcall_id1 = load_ast_nodes(node_file1)
    node_features2, func_nodes2, node_features_full2, node_names_full2, string_libcall_id2 = load_ast_nodes(node_file2)
    func_corpus1, func_lines1, func_corpus2, func_lines2 = find_unchanged_functions(corpus_file_new1, corpus_file_new2)
    unchanged_functions = set()
    matched_result = defaultdict(set)
    with open(matched_functions, "r") as f:
        func_list = f.readlines()
        matched1 = set()
        matched2 = set()
        for func_pair in func_list:
            func1 = func_pair.split(" ")[0]
            func2 = func_pair.split(" ")[1].strip()
            matched1.add(func1)
            matched2.add(func2)
            if func1 in func_corpus1 and func2 in func_corpus2:
                if func_lines1[func1] == func_lines2[func2]:
                    unchanged_functions.add((func1, func2))
                    for nodeid1, nodeid2 in zip(func_corpus1[func1], func_corpus2[func2]):
                        if nodeid1 != 'null' and nodeid2 != 'null':
                            matched_result[nodeid1].add(nodeid2)

    with open(training_node_path, "w") as f:
        matched_pair = []
        for func_pair in func_list:
            func1 = func_pair.split(" ")[0]
            func2 = func_pair.split(" ")[1].strip()
            if func1 in func_nodes1 and func2 in func_nodes2:
                for i in range(min(len(func_nodes1[func1]), len(func_nodes2[func2]))):
                    f.write(func_nodes1[func1][i] + " " + func_nodes2[func2][i] + "\n")
            # if func1 in func_summary1 and func2 in func_summary2:
            #     f.write(str(func_summary1[func1]) + " " + str(func_summary2[func2]) + "\n")

            # node text that are unique
            if func1 not in node_names_full1 or func2 not in node_names_full2:
                continue
            features1 = node_names_full1[func1]
            features2 = node_names_full2[func2]
            unique1 = set([i for i in features1.keys() if len(features1[i]) == 1])
            unique2 = set([i for i in features2.keys() if len(features2[i]) == 1])

            for feat in unique1.intersection(unique2):
                # f.write(list(features1[feat])[0] + " " + list(features2[feat])[0] + "\n")
                matched_pair.append((list(features1[feat])[0], list(features2[feat])[0]))

            # vsa values that are unique
            if func1 not in node_features1 or func2 not in node_features2:
                continue
            features1 = node_features1[func1]
            features2 = node_features2[func2]
            unique1 = set([i for i in features1.keys() if len(features1[i]) == 1])
            unique2 = set([i for i in features2.keys() if len(features2[i]) == 1])

            for feat in unique1.intersection(unique2):
                # f.write(list(features1[feat])[0] + " " + list(features2[feat])[0] + "\n")
                matched_pair.append((list(features1[feat])[0], list(features2[feat])[0]))

        for pair in matched_pair:
            i, j  = pair
            matched_result[i].add(j)

        # match the same string and libcalls
        for strlibcall in set(string_libcall_id1.keys()).intersection(set(string_libcall_id2.keys())):
            i = string_libcall_id1[strlibcall]
            j = string_libcall_id2[strlibcall]
            matched_result[i].add(j)

        for i in matched_result:
            if len(matched_result[i])==1:
                f.write(i + " " + list(matched_result[i])[0] + "\n")

        print(len(set(matched_pair)) / len(node_features_full1.keys()))

    if with_gt:
        map1 = get_source_lines(bin_path1, debug_info1, node_features_full1, base1)
        map2 = get_source_lines(bin_path2, debug_info2, node_features_full2, base2)
        add_attr_ast_nodes(node_file_new1, map1)
        add_attr_ast_nodes(node_file_new2, map2)
    else:
        add_attr_ast_nodes(node_file_new1, None)
        add_attr_ast_nodes(node_file_new2, None)


def process_two_files(bin_path1, bin_path2, output1, output2, compare_out, with_gt, func1_stripped=None, func2_stripped=None):
    # ----------------- 原有路径组装 ----------------
    filename1 = output1.split('/')[-1].split('_')[-1]
    filename2 = output2.split('/')[-1].split('_')[-1]
    node_file1 = os.path.join(output1, filename1+"_nodelabel.txt")
    node_file2 = os.path.join(output2, filename2+"_nodelabel.txt")
    image_base1 = os.path.join(output1, filename1+"_imagebase.txt")
    image_base2 = os.path.join(output2, filename2+"_imagebase.txt")
    edge_file1 = os.path.join(output1, filename1+"_edges.txt")
    edge_file2 = os.path.join(output2, filename2+"_edges.txt")
    corpus_file1 = os.path.join(output1, filename1+"_corpus.txt")
    corpus_file2 = os.path.join(output2, filename2+"_corpus.txt")
    debug_info1 = os.path.join(output1, filename1+"_debuginfo.txt")
    debug_info2 = os.path.join(output2, filename2+"_debuginfo.txt")
    matched_functions = os.path.join(compare_out, "matched_functions.txt")
    training_node_path = os.path.join(compare_out, "training_nodes.txt")
    # 为写入 compare_out 下的新文件准备路径
    dirname1 = output1.split('/')[-1]
    dirname2 = output2.split('/')[-1]
    node_file_new1 = os.path.join(compare_out, dirname1 + "_nodelabel.txt")
    node_file_new2 = os.path.join(compare_out, dirname2 + "_nodelabel.txt")
    edge_file_new1 = os.path.join(compare_out, dirname1 + "_edges.txt")
    edge_file_new2 = os.path.join(compare_out, dirname2 + "_edges.txt")
    corpus_file_new1 = os.path.join(compare_out, dirname1 + "_corpus.txt")
    corpus_file_new2 = os.path.join(compare_out, dirname2 + "_corpus.txt")

    # # -------------- 第一步：复制 nodelabel.txt 不变 （这是全流程） ---------------
    # shutil.copy(node_file1, node_file_new1)
    # shutil.copy(node_file2, node_file_new2)

    # —— 单函数模式：如果传了 func1/func2，就只保留那两个函数的子图
    if func1_stripped and func2_stripped:
        filter_to_function(node_file1, node_file_new1, func1_stripped)
        filter_to_function(node_file2, node_file_new2, func2_stripped)
    else:
        shutil.copy(node_file1, node_file_new1)
        shutil.copy(node_file2, node_file_new2)

    # 在写 edges.txt 前，加入「只保留子图内边」逻辑
    # 先从 node_file_new1 读出本子图的 ID 集合
    kept = set()
    with open(node_file_new1) as f:
        for L in f:
            if not L.startswith("#"):
                kept.add(L.split("|&|")[0])
    # 然后在写 base_edges 时只保留两端都在 kept 里的那些：
    def filter_edges(inf, outf):
        with open(inf) as fi, open(outf, "w") as fo:
            for l in fi:
                s,t,_ = l.strip().split(", ")
                if s in kept and t in kept:
                    fo.write(l)
    filter_edges(edge_file1, edge_file_new1)
    filter_edges(edge_file2, edge_file_new2)


    # ------------ 第二步：图结构增强 —— 添加跳转边 ----------
    # 1. 先载入每个节点的特征（从刚才复制的 nodelabel）
    nf1, map1 = load_node_features_and_mapping(node_file1)
    print(f"文件1nf1::[DEBUG] loaded {len(nf1)} nodes from {node_file1}")
    nf2, map2 = load_node_features_and_mapping(node_file2)
    print(f"文件2nf2::[DEBUG] loaded {len(nf2)} nodes from {node_file2}")
    # 2. 将原 edges + “跳转边” 写入新的 edges.txt
    jump1 = augment_edges_with_jumps(edge_file1, edge_file_new1, nf1, map1)
    # print(f"文件1jump1[DEBUG] wrote augmented edges to {edge_file_new1}, added {len(jump1)} jump‐edges")
    print(f"[DEBUG] 文件1添加了{len(jump1)} 跳转对")
    jump2 = augment_edges_with_jumps(edge_file2, edge_file_new2, nf2, map2)
    # print(f"文件1jump2[DEBUG] wrote augmented edges to {edge_file_new2}, added {len(jump2)} jump‐edges")
    print(f"[DEBUG] 文件2添加了{len(jump2)} 跳转对")

    # shutil.copy(edge_file1, edge_file_new1)
    # shutil.copy(edge_file2, edge_file_new2)


    # 特征级联前准备jumps.txt(分别产出 jumps1.txt / jumps2.txt)
    j1_path = os.path.join(compare_out, "jumps1.txt")
    with open(j1_path, "w") as jf1:
        # jump1, jump2 augment_edges_with_jumps 返回的 [(src_idx, tgt_idx), ...]
        for src_idx, tgt_idx in jump1:
            jf1.write(f"{src_idx} {tgt_idx}\n")

    j2_path = os.path.join(compare_out, "jumps2.txt")
    with open(j2_path, "w") as jf2:
        for src_idx, tgt_idx in jump2:
            jf2.write(f"{src_idx} {tgt_idx}\n")




    # -------------- 第三步：复制 corpus.txt 不变 --------------
    # shutil.copy(corpus_file1, corpus_file_new1)
    # shutil.copy(corpus_file2, corpus_file_new2)
    #  对 corpus.txt 也做相同 filter：

    def filter_corpus(inf, outf, func_name):
        with open(inf) as fi, open(outf,"w") as fo:
            keep=False
            for l in fi:
                if l.startswith("#"):
                    keep = (l[1:].strip()==func_name)
                    if keep: fo.write(l)
                elif keep:
                    fo.write(l)
    if func1_stripped and func2_stripped:
        filter_corpus(corpus_file1, corpus_file_new1, func1_stripped)
        filter_corpus(corpus_file2, corpus_file_new2, func2_stripped)
    else:
        shutil.copy(corpus_file1, corpus_file_new1)
        shutil.copy(corpus_file2, corpus_file_new2)



    # -------------- 其余保持不变 ----------------
    base1 = int(open(image_base1, "r").readline().strip())
    base2 = int(open(image_base2, "r").readline().strip())

    if with_gt:
        bin_path1 = bin_path1[:-8] # get rid of the stripped suffix
        bin_path2 = bin_path2[:-8]
    select_training_node(node_file1, node_file2, matched_functions, training_node_path, node_file_new1, node_file_new2, bin_path1, bin_path2, debug_info1, debug_info2, corpus_file_new1, corpus_file_new2, base1, base2, with_gt)

