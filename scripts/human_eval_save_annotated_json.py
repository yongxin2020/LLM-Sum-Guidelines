import json
import jsonlines
import pandas as pd
from collections import defaultdict

def get_ids_labels(annotation_file):
    ids, labels = [], []
    with jsonlines.open(annotation_file) as reader:
        for obj in reader:
            ids.append(obj["id"])
            labels.append(obj["label_annotations"])
    return ids, labels

def remove_irregular_sub_issues(label):
    if label in [{'A': '2'}, {'A': '1', 'B': '1', 'C': '1', 'D': '1', 'E': '1', 'F': '1'}, {'A': '1'}]:
        return None
    return label

def get_annotation_data(ids, labels, aspect, targets_mapper):
    data = []
    for idx, label in enumerate(labels):
        sample_id = ids[idx]
        
        if label is None:
            continue
            
        if aspect == "Sub-issues":
            # Handle Sub-issues specially
            if "Sub-issues" not in label:
                # Create entry with N/A for all systems if Sub-issues is missing
                Ordered_Dict = {
                    'reference': "N/A",
                    'BARThez': "N/A",
                    '3.5-Baseline': "N/A",
                    '3.5-G_O_A→B': "N/A",
                    '4-Baseline': "N/A",
                    '4-G_O_A→B': "N/A",
                    'id': sample_id
                }
                data.append(Ordered_Dict)
                continue
                
            sub_issues_label = label["Sub-issues"]
            filtered_label = remove_irregular_sub_issues(sub_issues_label)
            if filtered_label is None:
                # Mark as N/A if filtered out
                Ordered_Dict = {
                    'reference': "N/A",
                    'BARThez': "N/A",
                    '3.5-Baseline': "N/A",
                    '3.5-G_O_A→B': "N/A",
                    '4-Baseline': "N/A",
                    '4-G_O_A→B': "N/A",
                    'id': sample_id
                }
                data.append(Ordered_Dict)
                continue
                
            aspect_label = filtered_label
        else:
            aspect_label = label.get(aspect, None)
            if aspect_label is None:
                continue

        correct_models_list = [targets_mapper[m][sample_id] for m in ['A', 'B', 'C', 'D', 'E', 'F']]
        
        try:
            sample_label_new = dict(zip(correct_models_list, aspect_label.values()))
        except (AttributeError, ValueError):
            continue

        key_order = ['reference', 'BARThez', '3.5-Baseline', '3.5-G_O_A→B', '4-Baseline', '4-G_O_A→B']
        Ordered_Dict = {}
        for k in key_order:
            if k in sample_label_new:
                val = sample_label_new[k]
                if isinstance(val, dict):
                    scores = [int(v) for v in val.values() if v is not None]
                    Ordered_Dict[k] = sum(scores)/len(scores) if scores else "N/A"
                else:
                    try:
                        Ordered_Dict[k] = int(val) if val is not None else "N/A"
                    except (ValueError, TypeError):
                        Ordered_Dict[k] = "N/A"

        Ordered_Dict['id'] = sample_id
        data.append(Ordered_Dict)
    
    return pd.DataFrame(data)

def process_annotations(original_json_path, annotation_files, output_path, targets_mapper):
    with open(original_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    all_annotations = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for annotator_idx, ann_file in enumerate(annotation_files, start=1):
        print(f"Processing {ann_file}")
        for aspect in ["Faithfulness", "Main issues", "Sub-issues", "Resolution"]:
            ids, labels = get_ids_labels(ann_file)
            df = get_annotation_data(ids, labels, aspect, targets_mapper)
            
            for _, row in df.iterrows():
                sample_id = row['id']
                for system_id in ['reference', 'BARThez', '3.5-Baseline', '3.5-G_O_A→B', '4-Baseline', '4-G_O_A→B']:
                    if system_id in row:
                        val = row[system_id]
                        if pd.notna(val) and val != "N/A":  # Only store actual scores
                            all_annotations[sample_id][system_id][f"annotator{annotator_idx}"][aspect] = val
                        elif aspect == "Sub-issues":
                            # Explicitly store N/A for Sub-issues
                            all_annotations[sample_id][system_id][f"annotator{annotator_idx}"][aspect] = "N/A"

    enhanced_data = []
    for item in original_data:
        doc_id = item['doc_id']
        system_id = item['system_id']
        
        if doc_id in all_annotations and system_id in all_annotations[doc_id]:
            annotations = all_annotations[doc_id][system_id]
            
            aspect_sums = defaultdict(float)
            aspect_counts = defaultdict(int)
            overall_sum = 0
            overall_count = 0
            
            sub_issues_present = all("Sub-issues" in a and a["Sub-issues"] != "N/A" for a in annotations.values())
            
            for annotator, aspect_scores in annotations.items():
                for aspect, score in aspect_scores.items():
                    if score == "N/A":
                        continue
                    if aspect == "Sub-issues" and not sub_issues_present:
                        continue
                    try:
                        aspect_sums[aspect] += float(score)
                        aspect_counts[aspect] += 1
                        overall_sum += float(score)
                        overall_count += 1
                    except (ValueError, TypeError):
                        continue
            
            average_scores = {}
            for aspect in ["Faithfulness", "Main issues", "Sub-issues", "Resolution"]:
                if aspect_counts.get(aspect, 0) > 0:
                    if aspect == "Sub-issues" and not sub_issues_present:
                        continue
                    average_scores[aspect] = round(aspect_sums[aspect]/aspect_counts[aspect], 2)
            
            if overall_count > 0:
                average_scores['overall'] = round(overall_sum/overall_count, 2)
            
            enhanced_entry = {
                'doc_id': doc_id,
                'system_id': system_id,
                'dialogue': item['dialogue'],
                'reference': item['reference'],
                'system_output': item['system_output'],
                'annotations': annotations,
                'average_scores': average_scores
            }
            enhanced_data.append(enhanced_entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    print(f"Saved enhanced data to {output_path}")

if __name__ == "__main__":
    original_json_path = '../Guideline-Eval/data/decoda_eval_samples.json'
    with open("../results/human_annotations_decoda/targets_mapper_FR.json", "r", encoding="utf-8") as f:
        targets_mapper = json.load(f)
    annotation_files = [
        "../results/human_annotations_decoda/annotation_output/annotator_1_c/annotated_instances.jsonl",
        "../results/human_annotations_decoda/annotation_output/annotator_2_f/annotated_instances.jsonl",
        "../results/human_annotations_decoda/annotation_output/annotator_3_o/annotated_instances.jsonl"
    ]
    output_path = '../results/human_annotations_decoda/decoda_eval_samples_annotated.json'
    process_annotations(original_json_path, annotation_files, output_path, targets_mapper)