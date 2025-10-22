# evaluate.py
from collections import defaultdict
import os
import re
import traceback
import shutil
import numpy as np
import glob

def load_nodelabel(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    groundtruth = {}
    nodelines = {}
    funcname = None
    func_features = defaultdict(set)
    node2func = dict()
    func_nodes = {}
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            funcname = line[1:]
        else:
            features = line.split("|&|")
            groundtruth[features[0]] = features[-1]
            nodelines[features[0]] = features[1:]
            if funcname is not None:
                func_features[funcname].add(features[0])
                node2func[features[0]] = funcname
    return groundtruth, nodelines, node2func


def _safe_div(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


def evaluate_precision_recall_cross_version_token(out_dir, filepath1, filepath2, src_dir, filter=True):
    """
    Robust version: always returns a triple (prec, recall, f1).
    If no result files found or files empty -> returns (0.0, 0.0, 0.0)
    """
    try:
        filename1 = '_'.join(filepath1.split('/')[-2:])
        filename2 = '_'.join(filepath2.split('/')[-2:])
        comp_folder = filename1 + '_vs_' + filename2
        result_dir = os.path.join(out_dir, comp_folder + "_Finetuned-results")
        v1 = filename1.split("_")[0]
        v2 = filename2.split("_")[0]
        binary_name = filename1.split("_")[1]

        if not os.path.isdir(result_dir):
            print(f"[EVAL] result_dir not found: {result_dir} -- returning zeros")
            return (0.0, 0.0, 0.0)

        # search for files ending with suffix
        suffix = "-match_result.txt" if filter else "-Initial_match_result.txt"
        files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith(suffix)]
        if len(files) == 0:
            print(f"[EVAL] no files matching {suffix} under {result_dir} -- returning zeros")
            return (0.0, 0.0, 0.0)

        # We'll process any matching file (original logic processed first)
        for path in files:
            try:
                srclines1, nodefeatures1, node2func1 = load_nodelabel(os.path.join(out_dir, comp_folder, v1 + "_" + binary_name + "_nodelabel.txt"))
                srclines2, nodefeatures2, node2func2 = load_nodelabel(os.path.join(out_dir, comp_folder, v2 + "_" + binary_name + "_nodelabel.txt"))

                addrmap_path = os.path.join(out_dir, v1 + '_vs_' + v2 + '_addrMapping')
                currentdir = os.path.join(out_dir, comp_folder)

                if os.path.exists(addrmap_path):
                    shutil.copyfile(addrmap_path, os.path.join(currentdir, 'addrMapping'))
                    ground_truth = os.path.join(currentdir, 'addrMapping')
                elif not os.path.exists(os.path.join(currentdir, 'addrMapping')):
                    # attempt to generate via external script (best-effort)
                    groundTruthCollector = './gtc.py'
                    cmd_gtc = "python " + groundTruthCollector + \
                              ' --old_dir ' + os.path.join(src_dir, v1) + \
                              ' --new_dir ' + os.path.join(src_dir, v2) + \
                              ' --old_bin ' + filepath1[:-8] + \
                              ' --new_bin ' + filepath2[:-8] + \
                              ' --output_dir ' + currentdir
                    os.system(cmd_gtc)
                    if os.path.exists('addrMapping'):
                        shutil.copyfile('addrMapping', addrmap_path)
                        shutil.copyfile('addrMapping', os.path.join(currentdir, 'addrMapping'))
                        ground_truth = os.path.join(currentdir, 'addrMapping')
                    else:
                        print(f"[EVAL] addrMapping generation failed; continuing but mapping may be incomplete.")
                        ground_truth = os.path.join(currentdir, 'addrMapping')
                else:
                    ground_truth = os.path.join(currentdir, 'addrMapping')

                addrMapping = {}
                if os.path.exists(ground_truth):
                    with open(ground_truth) as f:
                        for line in f.readlines():
                            pair = re.findall(r'\[(.*?)\]', line)
                            if len(pair) == 2:
                                original_addr_list = pair[0].split(', ')
                                mod_addr_list = pair[1].split(', ')
                                for addr1 in original_addr_list:
                                    for addr2 in mod_addr_list:
                                        if len(addr1.split('/')) > 2:
                                            addr1 = '/'.join(addr1.split('/')[-2:])
                                        if len(addr2.split('/')) > 2:
                                            addr2 = '/'.join(addr2.split('/')[-2:])
                                        if addr1 not in addrMapping:
                                            addrMapping[addr1] = set()
                                        addrMapping[addr1].add(addr2)

                tp = 0
                fp = 0
                not_found = set()
                total1 = []

                with open(path, "r") as f:
                    match_results = [ln.strip() for ln in f.readlines() if ln.strip() != ""]

                if len(match_results) == 0:
                    # empty file -> return zeros
                    print(f"[EVAL] {path} empty -> returning zeros triple")
                    return (0.0, 0.0, 0.0)

                for line in match_results:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 6:
                        # skip malformed line
                        continue
                    n1, n2, gtline1, gtline2, correct, sim = parts[:6]
                    try:
                        simf = float(sim)
                    except Exception:
                        simf = 0.0
                    if gtline1 == "null":
                        continue
                    if simf < 0.1:
                        continue
                    if gtline1 not in addrMapping:
                        not_found.add(n1)
                        continue
                    toklst_field = None
                    if n1 in nodefeatures1:
                        # defensive: ensure length
                        if len(nodefeatures1[n1]) >= 4:
                            toklst_field = nodefeatures1[n1][-3]
                    if toklst_field is None:
                        # cannot compute tokens -> skip
                        continue
                    toklst = toklst_field.split('@*@')
                    num_tokens = len(toklst) // 2 if len(toklst) > 0 else 0
                    if gtline2 in addrMapping.get(gtline1, set()):
                        tp += num_tokens
                    else:
                        fp += num_tokens
                    total1.append(n1)

                total = 0
                for key in srclines1:
                    if key not in total1 and srclines1[key] not in addrMapping:
                        continue
                    if srclines1[key] != "null" and key not in not_found:
                        if key in nodefeatures1 and len(nodefeatures1[key]) >= 4:
                            toklst = nodefeatures1[key][-3].split('@*@')
                            num_tokens = len(toklst) // 2 if len(toklst) > 0 else 0
                            total += num_tokens

                prec = _safe_div(tp, (tp + fp))
                recall = _safe_div(tp, total)
                f1 = _safe_div(2 * prec * recall, (prec + recall)) if (prec + recall) > 0 else 0.0
                print(binary_name, tp, fp, total, recall, prec, f1)
                return (prec, recall, f1)
            except Exception:
                print(f"[EVAL] error processing file {path}:")
                print(traceback.format_exc())
                # try next file
                continue

        # If we tried all files but none produced a result, return zeros
        print(f"[EVAL] processed all files but none yielded result -> returning zeros")
        return (0.0, 0.0, 0.0)
    except Exception:
        print(traceback.format_exc())
        return (0.0, 0.0, 0.0)


def evaluate_precision_recall_cross_optlevel_token(out_dir, filepath1, filepath2, filter=True):
    """
    Robust version for optlevel evaluation. Returns (prec, recall, f1) with safe fallbacks.
    """
    try:
        filename1 = '_'.join(filepath1.split('/')[-2:])
        filename2 = '_'.join(filepath2.split('/')[-2:])
        comp_folder = filename1 + '_vs_' + filename2
        result_dir = os.path.join(out_dir, comp_folder + "_Finetuned-results")
        v1 = filename1.split("_")[0]
        v2 = filename2.split("_")[0]
        binary_name = filename1.split("_")[1]

        if not os.path.isdir(result_dir):
            print(f"[EVAL] result_dir not found: {result_dir} -- returning zeros")
            return (0.0, 0.0, 0.0)

        suffix = "-match_result.txt" if filter else "-Initial_match_result.txt"
        files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith(suffix)]
        if len(files) == 0:
            print(f"[EVAL] no files matching {suffix} under {result_dir} -- returning zeros")
            return (0.0, 0.0, 0.0)

        # loop files
        for path in files:
            try:
                srclines1, nodefeatures1, node2func1 = load_nodelabel(os.path.join(out_dir, comp_folder, v1 + "_" + binary_name + "_nodelabel.txt"))
                srclines2, nodefeatures2, node2func2 = load_nodelabel(os.path.join(out_dir, comp_folder, v2 + "_" + binary_name + "_nodelabel.txt"))

                srclines_set2 = set(srclines2.values())

                tp = 0
                fp = 0
                not_found = set()
                tp_set = set()
                fp_set = set()
                matched = set()

                with open(path, "r") as f:
                    match_results = [ln.strip() for ln in f.readlines() if ln.strip() != ""]

                if len(match_results) == 0:
                    print(f"[EVAL] {path} empty -> returning zeros")
                    return (0.0, 0.0, 0.0)

                for line in match_results:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 6:
                        continue
                    n1, n2, gtline1, gtline2, correct, sim = parts[:6]
                    try:
                        simf = float(sim)
                    except Exception:
                        simf = 0.0

                    if n1 in matched and int(n1) < 100:
                        tp_set.clear()
                        fp_set.clear()
                        matched.clear()
                    else:
                        matched.add(n1)

                    if simf < 0.1:
                        continue
                    if gtline1 == "null":
                        continue
                    if gtline1 not in srclines_set2:
                        not_found.add(n1)
                        continue
                    if gtline1 == gtline2:
                        tp_set.add(n1)
                    else:
                        fp_set.add(n1)

                for n1 in tp_set:
                    if n1 in nodefeatures1 and len(nodefeatures1[n1]) >= 4:
                        toklst = nodefeatures1[n1][-3].split('@*@')
                        num_tokens = len(toklst) // 2 if len(toklst) > 0 else 0
                        tp += num_tokens

                for n1 in fp_set:
                    if n1 in nodefeatures1 and len(nodefeatures1[n1]) >= 4:
                        toklst = nodefeatures1[n1][-3].split('@*@')
                        num_tokens = len(toklst) // 2 if len(toklst) > 0 else 0
                        fp += num_tokens

                total = 0
                for key in srclines1:
                    if srclines1[key] != "null" and key not in not_found:
                        if key in nodefeatures1 and len(nodefeatures1[key]) >= 4:
                            toklst = nodefeatures1[key][-3].split('@*@')
                            num_tokens = len(toklst) // 2 if len(toklst) > 0 else 0
                            total += num_tokens

                prec = _safe_div(tp, (tp + fp))
                recall = _safe_div(tp, total)
                f1 = _safe_div(2 * prec * recall, (prec + recall)) if (prec + recall) > 0 else 0.0
                print(binary_name, tp, fp, total, prec, recall, f1)
                return (prec, recall, f1)
            except Exception:
                print(f"[EVAL] error processing file {path}:")
                print(traceback.format_exc())
                continue

        print(f"[EVAL] processed all files but none yielded result -> returning zeros")
        return (0.0, 0.0, 0.0)
    except Exception:
        print(traceback.format_exc())
        return (0.0, 0.0, 0.0)


if __name__ == "__main__":
    # example (no-op)
    print("evaluate.py: run via import by sigmadiff.py")
