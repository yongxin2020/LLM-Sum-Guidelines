import json
import pandas as pd
import jsonlines
from compute_metrics import decoda_read_jsonlines_file
import nltk
import os
import numpy as np

# Load mapping and annotation files
targets_mapper = json.load(open("../results/human_annotations_decoda/targets_mapper_FR.json"))
annotation_files = [
    "../results/human_annotations_decoda/annotation_output/annotator_3_o/annotated_instances.jsonl",
    "../results/human_annotations_decoda/annotation_output/annotator_2_f/annotated_instances.jsonl",
    "../results/human_annotations_decoda/annotation_output/annotator_1_c/annotated_instances.jsonl"
]
test_file = "../data/decoda/test.json"

# Prepare dialogue and ids
dialogue_list, synopsis_list, ids = decoda_read_jsonlines_file(test_file)
with jsonlines.open(test_file) as reader:
    df_decoda = pd.DataFrame.from_dict(reader, orient='columns')
ids_100 = list(set(ids))

def get_summ_length(predictions):
    return [len(nltk.word_tokenize(str(summary))) for summary in predictions]

dialogue_length = get_summ_length([df_decoda.loc[df_decoda['id'] == id].iloc[0]['dialogue'] for id in ids_100])

def get_top_indices(lengths, n=10, largest=True):
    arr = pd.Series(lengths)
    return arr.nlargest(n).index.tolist() if largest else arr.nsmallest(n).index.tolist()

max_index = get_top_indices(dialogue_length, 10, largest=True)
min_index = get_top_indices(dialogue_length, 10, largest=False)
dial_max_ids = [ids_100[i] for i in max_index]
dial_min_ids = [ids_100[i] for i in min_index]

def get_ids_labels(annotation_file):
    ids, labels = [], []
    with jsonlines.open(annotation_file) as reader:
        for obj in reader:
            ids.append(obj["id"])
            labels.append(obj["label_annotations"])
    return ids, labels

def extract_labels(labels, key):
    # Always return a dict or None
    return [label.get(key, None) if isinstance(label, dict) else None for label in labels]

def remove_irregular_sub_issues(labels_Sub_issues):
    for i in range(len(labels_Sub_issues)):
        if labels_Sub_issues[i] in [{'A': '2'}, {'A': '1', 'B': '1', 'C': '1', 'D': '1', 'E': '1', 'F': '1'}, {'A': '1'}]:
            labels_Sub_issues[i] = None
    return labels_Sub_issues

def get_annotation_data(ids, labels, aspect):
    data = []
    for idx, label in enumerate(labels):
        sample_id = ids[idx]
        if aspect == "Sub_issues" and label is None:
            continue
        if aspect == "Sub_issues":
            label = remove_irregular_sub_issues([label])[0]
            if label is None:
                continue
            correct_models_list = [targets_mapper[m][sample_id] for m in ['A', 'B', 'C', 'D', 'E', 'F']]
            sample_label_new = dict(zip(correct_models_list, label.values()))
        else:
            # For other aspects, label is already a dict mapping model to value
            correct_models_list = [targets_mapper[m][sample_id] for m in ['A', 'B', 'C', 'D', 'E', 'F']]
            sample_label_new = dict(zip(correct_models_list, label.values()))
        key_order = ['reference', 'BARThez', '3.5-Baseline', '3.5-G_O_A→B', '4-Baseline', '4-G_O_A→B']
        Ordered_Dict = {k: int(sample_label_new[k]) for k in key_order}
        Ordered_Dict['id'] = sample_id
        data.append(Ordered_Dict)
    df = pd.DataFrame(data)
    df_min = df[df['id'].isin(dial_min_ids)]
    df_max = df[df['id'].isin(dial_max_ids)]
    return df, df_min, df_max

def get_average(df):
    cols = ['reference', 'BARThez', '3.5-Baseline', '3.5-G_O_A→B', '4-Baseline', '4-G_O_A→B']
    means = [round(df[c].mean(), 2) for c in cols]
    stds = [round(df[c].std(), 2) for c in cols]
    return means, stds

def process_aspect(aspect, annotation_files):
    dfs, dfs_min, dfs_max = [], [], []
    for ann_file in annotation_files:
        ids, labels = get_ids_labels(ann_file)
        if aspect == "Sub_issues":
            labels = extract_labels(labels, "Sub-issues")
        else:
            labels = extract_labels(labels, aspect)
        df, df_min, df_max = get_annotation_data(ids, labels, aspect)
        dfs.append(df)
        dfs_min.append(df_min)
        dfs_max.append(df_max)
    df_all = pd.concat(dfs, axis=0)
    df_min_all = pd.concat(dfs_min, axis=0)
    df_max_all = pd.concat(dfs_max, axis=0)
    means, stds = get_average(df_all)
    means_min, stds_min = get_average(df_min_all)
    means_max, stds_max = get_average(df_max_all)
    return (means, stds), (means_min, stds_min), (means_max, stds_max)

def save_results_to_csv(results, aspects, out_path):
    model_cols = ['Reference', 'BARThez', '3.5-WL', '3.5-HGR→WL', '4-WL', '4-HGR→WL']
    rows = []
    for aspect, ((means, stds), (means_min, stds_min), (means_max, stds_max)) in zip(aspects, results):
        for split, m, s in zip(['All (20)', '10 shortest', '10 longest'], [means, means_min, means_max], [stds, stds_min, stds_max]):
            row = {'Aspect': aspect, 'Split': split}
            for col, mean_val, std_val in zip(model_cols, m, s):
                row[f"{col} Mean"] = mean_val
                row[f"{col} Std"] = std_val
            rows.append(row)
    df = pd.DataFrame(rows)
    interleaved = ['Aspect', 'Split']
    for col in model_cols:
        interleaved.append(f"{col} Mean")
        interleaved.append(f"{col} Std")
    df = df[interleaved]

    df = df[interleaved]
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    aspects = ["Faithfulness", "Main issues", "Sub_issues", "Resolution"]
    results = [process_aspect(aspect, annotation_files) for aspect in aspects]
    save_results_to_csv(results, aspects, out_path="../results/human_eval_summary.csv")

