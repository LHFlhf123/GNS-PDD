# import os用于文件和目录操作(判断路径、创建文件夹、执行系统命令等)
import os
# 引入命令行参数解析的工具 ，ArgumentParser：创建解析器；ArgumentDefaultsHelpFormatter：自动在帮助里显示默认值；BooleanOptionalAction：让 --with_gt / --no-with_gt 这种布尔切换更方便
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
# 导入函数级 diff 的核心逻辑
from diffing import diff_two_files
#  导入从 diff 结果中选取训练节点的函数
from choose_train_nodes import process_two_files
import shutil
#  用于修改 sys.path，方便引入非标准路径下的模块
import sys
import time
from load_emb import Function
#  导入两种评估方式：一种针对同版本不同优化级别（optlevel），另一种针对跨版本 diff。
from evaluate import evaluate_precision_recall_cross_optlevel_token, evaluate_precision_recall_cross_version_token
#  将子目录 dgmc_batch_decompile 加入 Python 模块搜索路径
sys.path.append('dgmc_batch_decompile')
# from TestOwnDataUseModel import processDGMC
#  从 DGMC模型的封装脚本中，导入批量运行 DGMC模型的函数
from dgmc_batch_decompile.TestOwnDataUseModel import processDGMC
from dgmc_batch_decompile.TestOwnDataUseModel import build_func_maps

# 第一步的提取特征
def extract_features(filepath, output, ghidra_home, ghidra_proj_name, with_gt=True):
    current_dir = os.path.dirname(os.path.realpath(__file__))  # 取到脚本自身所在目录的绝对路径
    script_dir = os.path.join(current_dir, 'ghidra_script')  # Ghidra Java 脚本所在的子目录 ghidra_script
    tmp_dir = os.path.join(current_dir, 'tmp')  #  用于 Ghidra headless 生成中间项目的临时文件夹
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if not os.path.exists(output):
        os.makedirs(output)
    elif os.path.exists(output+'/addr2funcname.txt'):  # 如果已存在 addr2funcname.txt，说明已经跑过一次，就直接返回（跳过重复处理）
        return
    #  在输出目录下再建一个子文件夹 decompiled，用来存放 Ghidra 导出的反编译结果。
    if not os.path.exists(os.path.join(output, 'decompiled')):
        os.makedirs(os.path.join(output, 'decompiled'))

    # get ground truth  有 ground truth 时，先跑 Ghidra 的 CollectGroundTruth.java 脚本
    if with_gt:
        # filepath[:-8]：假设输入文件名带上了后缀如 xxx_cmpstripped，这里去掉最后 8 个字符，找到原始二进制。
        # -postScript CollectGroundTruth.java：收集 ground truth（如函数地址到函数名映射），CollectGroundTruth.java：这个脚本会把地址→函数名的映射输出到 output 目录下
        vsa_command = ghidra_home + "/support/analyzeHeadless " + tmp_dir + " " + ghidra_proj_name + " -import " + filepath[:-8] + " -overwrite -scriptPath " + script_dir + " -postScript CollectGroundTruth.java " + output
        # print(vsa_command)
        # os.system 执行这条命令，调用 Ghidra headless 模式运行 CollectGroundTruth.java
        os.system(vsa_command)

    # run preprocessing  再跑核心预处理脚本 VSAPCode.java
    print("run scripts: " + time.strftime("%H:%M:%S", time.localtime()))
    # -postScript VSAPCode.java 这是 SigmaDiff 的核心预处理脚本，它会在 Ghidra 中遍历所有函数，提取函数的反编译伪代码、控制流图、数据流图、汇编指令序列、基本块信息等，将这些特征格式化成供后续 diff 和图匹配模块使用的文件，通常包括 JSON、GraphML、文本 Token 列表等
    # 把所有预处理结果输出到同一个 output 目录，与 ground truth 放一起
    vsa_command = ghidra_home + "/support/analyzeHeadless " + tmp_dir + " " + ghidra_proj_name + " -import " + filepath + " -overwrite -scriptPath " + script_dir + " -postScript VSAPCode.java " + output
    # print(vsa_command)
    os.system(vsa_command)

    # 开始进行比较两个二进制文件
def compare_two_bins(filepath1, filepath2, args):
    with_gt = args.with_gt
    output_dir = args.output_dir
    ghidra_home = args.ghidra_home
    src_dir = args.src_dir
    ghidra_proj_name = args.ghidra_proj_name

    # 用 strip（剥离符号表）生成一个新的 stripped 文件，并在后面拼接 "stripped" 作为新文件名，以保证后续分析只用剥过符号的二进制。
    if with_gt:
        if "arm" in filepath1:
            os.system("arm-linux-gnueabi-strip -s " + filepath1 + " -o " + filepath1 + "stripped")
        else:
            # 符号表：就是函数名/地址映射还在，通过 Ghidra 的 CollectGroundTruth.java 可以把这些“真值”提取出来，后续评估模型的匹配结果时，就有标准答案可以对比。
            os.system("strip -s " + filepath1 + " -o " + filepath1 + "stripped")
        if "arm" in filepath2:
            os.system("arm-linux-gnueabi-strip -s " + filepath2 + " -o " + filepath2 + "stripped")
        else:
            os.system("strip -s " + filepath2 + " -o " + filepath2 + "stripped")
        #  把原来的路径字符串后面拼接 "stripped"，也就是让后续所有对 filepath1、filepath2 的引用都指向新生成的剥离版本。
        filepath1 += "stripped"
        filepath2 += "stripped"

    # 生成唯一且可识别的输出子目录名
    filename1 = '_'.join(filepath1.split('/')[-2:]) # e.g., diffutils-2.8-O0_cmpstripped
    filename2 = '_'.join(filepath2.split('/')[-2:])  #  取路径最后两级 目录/文件名，把它们用下划线拼成一段，作为目录名

    # 构建各阶段输出目录
    output1 = os.path.join(output_dir, filename1)  # 用来存放第一个二进制（剥符号后）的特征提取结果
    output2 = os.path.join(output_dir, filename2)
    compare_out = os.path.join(output_dir, filename1 + '_vs_' + filename2)  #  两者 diff、训练节点、DGMC 输出等综合结果的目录，命名成 <filename1>_vs_<filename2>
    if not os.path.exists(output1):
        os.makedirs(output1)
    if not os.path.exists(output2):
        os.makedirs(output2)

    #  记录开始时间，用于统计单对比较的耗时。
    t0 = time.time()

    # 第一步预处理preprocess， 即对两个二进制文件提取特征
    extract_features(filepath1, output1, ghidra_home, ghidra_proj_name, with_gt)
    extract_features(filepath2, output2, ghidra_home, ghidra_proj_name, with_gt)

    # # 第二步在两组特征上做函数级diff function level diffing
    # diff_two_files(output1, output2, compare_out, with_gt)
    #
    # # 第三步根据diff结果，从图中选出训练节点(正负样本对)choose training node
    # process_two_files(filepath1, filepath2, output1, output2, compare_out, with_gt)


    # 改(只分析两个函数)    如果给了 func1/func2，就只比较那两个函数，不走全程序调⽤图
    if args.func1 and args.func2:
        print(f"[FUNC-MODE] 只对函数 {args.func1} vs {args.func2} 继续后续流程")
        # diff_two_files 不变，用 matched_functions.txt 驱动单函数训练节点选取
        diff_two_files(output1, output2, compare_out, with_gt)
        # stripped_map = build_func_maps(compare_out)
        # print("[DEBUG] stripped_map mapping:", stripped_map)

        # —— 从各自的特征目录读取 addr2funcname 映射 ——
        map1 = build_func_maps(output1)
        map2 = build_func_maps(output2)
        print(f"[DEBUG] func→stripped @ output1: {map1}")
        print(f"[DEBUG] func→stripped @ output2: {map2}")
        f1s = map1.get(args.func1)
        f2s = map2.get(args.func2)
        if f1s is None or f2s is None:
            raise RuntimeError(f"无法映射 {args.func1}/{args.func2} 到剥符号名，请检查 addr2funcname.txt")
        # 这里 process_two_files 会直接载入 matched_functions.txt 里指定的那对函数对应节点
        process_two_files(filepath1, filepath2, output1, output2, compare_out, with_gt, func1_stripped=f1s, func2_stripped=f2s)
    else:
        # 旧流程：先 match 调⽤图、然后 diff
        diff_two_files(output1, output2, compare_out, with_gt)
        process_two_files(filepath1, filepath2, output1, output2, compare_out, with_gt, None, None)




    # # 第四步用 Deep Graph Matching Consensus 模型，做图匹配推理，输出匹配结果。 run DGMC model
    # processDGMC(output_dir, filename1, filename2, args)

    # processDGMC(output_dir, filename1, filename2, args, func1=f1s, func2=f2s)
    # — 运行 DGMC —
    if args.func1 and args.func2:
        processDGMC(output_dir, filename1, filename2, args, func1=f1s, func2=f2s)
    else:
        processDGMC(output_dir, filename1, filename2, args, func1=None, func2=None)



    # 记录当前对比总耗时
    total_time = time.time() - t0

    # 第五步，进行评估操作  evaluate
    if with_gt:
        # 因为上面在filename1上加了唯一且可识别的输出子目录，也就是将路径最后两级(目录名和文件名)用_拼接过
        version1 = filename1.split('_')[0].split('-')[1]  # 例：coreutils-8.1-clang_[ version1 = 8.1
        version2 = filename2.split('_')[0].split('-')[1]
        if version1 == version2:
            if "arm" in filename1:
                filtered = False
            else:
                filtered = True
            prec, recall, f1 = evaluate_precision_recall_cross_optlevel_token(output_dir, filename1, filename2, filtered)
        else:
            if src_dir is None:
                print("the directory of source code is needed for cross-version evaluation")
                return None
            prec, recall, f1 = evaluate_precision_recall_cross_version_token(output_dir, filepath1, filepath2, src_dir)

        # 记录时间和F1(每个函数的F1)
        f = open(os.path.join(args.output_dir, 'time.txt'), 'a')
        size2 = os.path.getsize(filepath2[:-8])
        size1 = os.path.getsize(filepath1[:-8])
        size = (size1+size2)/2
        # time.txt中记录的本次对比的耗时、二进制大小（平均）、F1
        f.write(','.join([filename1 + '_vs_' + filename2, str(total_time), str(size), str(f1)]) + '\n')
        f.close()
        # 返回指标，供批量模式聚合使用。
        return prec, recall, f1, total_time


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--input1', required=True, help='The path of input bin file 1 or a group of bin files')
    parser.add_argument('--input2', required=True, help='The path of input bin file 2 or a group of bin files')
    parser.add_argument('--with_gt', required=False, action=BooleanOptionalAction, help='whether the input has ground truth or not, True if add this option, False if use --no-with_gt')
    parser.add_argument('--src_dir', required=False, help='The home directory of source code, used for cross-version diffing evaluation')
    parser.add_argument('--ghidra_home', required=True, help='Home directory of Ghidra')
    parser.add_argument('--output_dir', required=True, help='Specify the output directory')
    parser.add_argument('--ghidra_proj_name', required=True)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--rnd_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--k', type=int, default=25)
    parser.add_argument('--in_channels', type=int, default=128)

    # 改--只比较函数
    parser.add_argument('--func1', type=str, help='The first function name to be compared')
    parser.add_argument('--func2', type=str, help='The second function name to be compared')
    args = parser.parse_args()

    path1 = args.input1
    path2 = args.input2

    # diff a group of binaries 差分一对二进制文件目录时
    if os.path.isdir(path1) and os.path.isdir(path2):
        prec_average = []
        recall_average = []
        f1_average = []
        time_average = []
        #  聚合所有对比的平均 Prec/Recall/F1/Time，写入 finalresults.txt
        f = open(os.path.join(args.output_dir, 'finalresults.txt'), 'a')
        path2bin = set(os.listdir(path2))
        for binary in os.listdir(path1):
            # 判定两个文件夹中相同文件名的二进制文件进行操作
            if not binary.endswith("stripped") and binary in path2bin:
                ret = compare_two_bins(os.path.join(path1, binary), os.path.join(path2, binary), args)
                if ret is not None:
                    prec, recall, f1, t = ret
                    prec_average.append(prec)
                    recall_average.append(recall)
                    f1_average.append(f1)
                    time_average.append(t)
        # 打印并写入finalresults.txt 平均指标(这里在运行run_test.sh是注意里面的echo "version, precision, recall, f1-score, avg_time" >> $home/out_arch/finalresults.txt参数，特别是out_arch路径别忘了改)
        print(path1.split('/')[-1] +'_vs_' + path2.split('/')[-1], sum(prec_average)/len(prec_average), sum(recall_average)/len(recall_average), sum(f1_average)/len(f1_average))
        f.write(','.join([path1.split('/')[-1] +'_vs_' + path2.split('/')[-1], str(sum(prec_average)/len(prec_average)), str(sum(recall_average)/len(recall_average)), str(sum(f1_average)/len(f1_average)), str(sum(time_average)/len(time_average))]) + '\n')
        f.close()

    # diff two binaries  差分二个二进制文件
    if os.path.isfile(path1) and os.path.isfile(path2):
        precision1, recall1, F1, t1 = compare_two_bins(path1, path2, args)
        print("-------------------最终指标为:\n")

        name1 = '_'.join(path1.split('/')[-2:])
        name2 = '_'.join(path2.split('/')[-2:])
        print(f"{name1}_vs{name2}", precision1, recall1 , F1 , t1)

        print("---------------------执行结束------------")