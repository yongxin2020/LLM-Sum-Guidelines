"""
For DialogSum (3 references per dialogue), we compare model outputs against human variance:
ROUGE and BERTScore
"""

import numpy as np
import pandas as pd
from evaluate import load
from compute_metrics import read_csv_to_df, read_txt_to_list, read_jsonlines_file, get_predictions
import argparse
import logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress "Using default tokenizer" messages

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='decoda', help="The name of the dataset: dialogsum, decoda")
parser.add_argument('--test_file', default='../data/dialogsum/dialogsum.test.jsonl', help="Path to the DialogSum test file")
parser.add_argument('--prompt_type', default='Baseline', help="OneStep: Baseline, Guideline_Original, Guideline_Original_Annotator; TwoStep: Guideline_Original_ToBaseline, Guideline_Original_Annotator_ToBaseline")
parser.add_argument('--model', default='gpt-3.5', help="gpt-3.5, gpt-4")

# Load the ROUGE and BERTScore metrics from the Hugging Face evaluate library
rouge = load("rouge")
bertscore = load("bertscore")

def compute_rouge(predictions_list, references_list):
    result = rouge.compute(predictions=[predictions_list], references=[references_list], use_stemmer=True)

    # Return ROUGE-1, ROUGE-2, and ROUGE-L scores (in percentage form)
    return round(result["rouge1"] * 100, 2), round(result["rouge2"] * 100, 2), round(result["rougeL"] * 100, 2)

def compute_bertscore(reference1, reference2, lang="en"):
    scores = bertscore.compute(predictions=[reference1], references=[reference2], lang=lang)
    return scores["f1"][0]  # Extract F1-score

def analyze_variation(summary1_list, summary2_list, summary3_list, model_predictions, lang="en"):
    """
    Computes the variation among reference summaries using ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore,
    and checks if model predictions fall within the reference distribution.
    """

    rouge1_variations, rouge2_variations, rougeL_variations = [], [], []
    bert_variations = []

    for s1, s2, s3 in zip(summary1_list, summary2_list, summary3_list): # Compute one sentence with one other sentence each time
        # Compute ROUGE between reference summaries
        rouge_scores = [
            compute_rouge(s1, s2),
            compute_rouge(s1, s3),
            compute_rouge(s2, s3)
        ]
        print("rouge_scores:", rouge_scores)

        # Store separate ROUGE scores
        rouge1_variations.extend([r[0] for r in rouge_scores])  # ROUGE-1
        rouge2_variations.extend([r[1] for r in rouge_scores])  # ROUGE-2
        rougeL_variations.extend([r[2] for r in rouge_scores])  # ROUGE-L

        # Compute BERTScore between reference summaries
        bert_scores = [
            compute_bertscore(s1, s2, lang),
            compute_bertscore(s1, s3, lang),
            compute_bertscore(s2, s3, lang)
        ]
        print("bert_scores:", bert_scores)
        bert_variations.extend(bert_scores)

    # Compute mean and standard deviation of reference variation
    reference_stats = {
        "ROUGE-1": (np.mean(rouge1_variations), np.std(rouge1_variations)),
        "ROUGE-2": (np.mean(rouge2_variations), np.std(rouge2_variations)),
        "ROUGE-L": (np.mean(rougeL_variations), np.std(rougeL_variations)),
        "BERTScore": (np.mean(bert_variations), np.std(bert_variations))
    }

    print("Reference Variance Statistics:")
    for key, (mean, std) in reference_stats.items():
        print(f"{key} Mean: {mean:.4f}, Std Dev: {std:.4f}")

    # Compare model predictions against reference variance
    model_rouge1_diffs, model_rouge2_diffs, model_rougeL_diffs, model_bert_diffs = [], [], [], []

    for pred, s1, s2, s3 in zip(model_predictions, summary1_list, summary2_list, summary3_list):
        # Compute model ROUGE vs references
        model_rouge_scores = [
            compute_rouge(pred, s1),
            compute_rouge(pred, s2),
            compute_rouge(pred, s3)
        ]

        # Store separate ROUGE scores
        model_rouge1_diffs.append(np.mean([r[0] for r in model_rouge_scores]))  # ROUGE-1
        model_rouge2_diffs.append(np.mean([r[1] for r in model_rouge_scores]))  # ROUGE-2
        model_rougeL_diffs.append(np.mean([r[2] for r in model_rouge_scores]))  # ROUGE-L

        # Compute model BERTScore vs references
        model_bert_scores = [
            compute_bertscore(pred, s1, lang),
            compute_bertscore(pred, s2, lang),
            compute_bertscore(pred, s3, lang)
        ]
        model_bert_diffs.append(np.mean(model_bert_scores))  # Avg model-reference BERTScore

    # Compute model mean ROUGE and BERTScore
    model_stats = {
        "ROUGE-1": np.mean(model_rouge1_diffs),
        "ROUGE-2": np.mean(model_rouge2_diffs),
        "ROUGE-L": np.mean(model_rougeL_diffs),
        "BERTScore": np.mean(model_bert_diffs)
    }

    print("\nModel Performance:")
    for key, mean in model_stats.items():
        print(f"{key} Mean: {mean:.4f}")

    # Check if model scores fall within reference variance
    model_in_variance = {
        metric: (reference_stats[metric][0] - reference_stats[metric][1]) <= model_stats[metric] <= (
                    reference_stats[metric][0] + reference_stats[metric][1])
        for metric in reference_stats.keys()
    }

    for metric, in_variance in model_in_variance.items():
        print(f"Model {metric} inside reference variance? {'Yes' if in_variance else 'No'}")

    return {
        "Reference Stats": reference_stats,
        "Model Stats": model_stats,
        "Model in Variance": model_in_variance
    }

def select_first_of_every_three(lst):
    return lst[::3]  # Select every third element starting from index 0

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    prompt_type = args.prompt_type
    model = args.model
    test_file = args.test_file

    # These three prompts generate better results for GPT models
    OneStep = ["Baseline", "Guideline_Original_Annotator"]
    TwoStep = ["Guideline_Original_Annotator_ToBaseline"]

    if dataset == "dialogsum":
        dialogue_list, summary1_list, summary2_list, summary3_list, ids = read_jsonlines_file(test_file) # len(summary1_list): 500

        if model == "bart-based":
            predictions = read_txt_to_list("../dialogsum/bart_large_generated_summaries.txt")
            print("bart_large_predictions", model)
            model_predictions = select_first_of_every_three(predictions)
            result = analyze_variation(summary1_list, summary2_list, summary3_list, model_predictions, lang="en")
        else:
            model_predictions = get_predictions(prompt_type, dataset, model)
            print(prompt_type, model)
            result = analyze_variation(summary1_list, summary2_list, summary3_list, model_predictions, lang="en")
    else:
        raise ValueError(f"Unknown dataset: {dataset}") 
