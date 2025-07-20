"""
check the data points on which the GPT-generate summaries achieve low ROUGE but high BERTScore scores. 
return top 10 samples (reference/prediction) with the greatest discrepancy

4-WL dialogsum (EN) and decoda (FR)
prompt_type: Baseline (WL)

rougeL and bertscore (F1):
- If you want to prioritize accuracy (conciseness and relevance of predictions), go with precision.
- If you want to prioritize coverage (ensuring predictions capture all relevant points), go with recall.
- If you want a balance between the two, stick with F1.
functions calculate_rouge_bertscore_multireferences() not yet correct, I want to have the average score of three references to compare with the prediction
"""

import evaluate
import pandas as pd
import argparse
import os
from compute_metrics import (
    read_jsonlines_file, get_predictions, combine_ref_pred,
    decoda_read_jsonlines_file, read_csv_to_df
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dialogsum', help="Dataset: dialogsum or decoda")
parser.add_argument('--prompt_type', default='Baseline', help="Prompt type")
parser.add_argument('--model', default='gpt-4', help="Model: gpt-3.5, gpt-4")
parser.add_argument('--test_file', default=None, help="Path to test file")
parser.add_argument('--output_dir', default='../results/examples_analysis_rouge_bertscore', help="Directory to save results")
parser.add_argument('--top_k', type=int, default=10, help="Number of top examples to show")

# Function to compute ROUGE at the sample level
def compute_rouge(predictions_list, references_list):
    metric = evaluate.load("rouge")
    # Compute ROUGE scores for each pair of prediction and reference
    scores = []
    for prediction, reference in zip(predictions_list, references_list):
        result = metric.compute(predictions=[prediction], references=[reference], use_stemmer=True)
        # Extract scalar ROUGE scores directly
        rouge1 = result["rouge1"]
        rouge2 = result["rouge2"]
        rougeL = result["rougeL"]
        rougeLsum = result["rougeLsum"]
        scores.append({"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL, "rougeLsum": rougeLsum})
    return scores

# Function to compute BERTScore at the sample level
def compute_bertscore(predictions_list, references_list, lang="en"):
    metric = evaluate.load("bertscore")
    results = metric.compute(predictions=predictions_list, references=references_list, lang=lang)
    # Extract F1 scores (BERTScore) for each example
    return results["f1"]

# Function to calculate ROUGE and BERTScore and find examples with the largest difference
def calculate_rouge_bertscore(predictions, references, lang="en", k=10, output_fp=None):
    # Compute ROUGE scores for all examples
    rouge_scores = compute_rouge(predictions, references)
    # Compute BERTScore F1 for all examples
    bert_scores = compute_bertscore(predictions, references, lang=lang)

    # Prepare a list to hold all examples with their scores
    examples = []
    for i, (prediction, reference) in enumerate(zip(predictions, references)):
        rougeL = rouge_scores[i]["rougeL"]  # Use ROUGE-L as the representative ROUGE score
        bert_score = bert_scores[i]
        difference = bert_score - rougeL  # Calculate the difference
        examples.append({
            "id": i,  # Keep the original ID
            "reference": reference,
            "prediction": prediction,
            "ROUGE": round(rougeL, 4),
            "BERTScore": round(bert_score, 4),
            "Difference": round(difference, 4),
        })

    # Sort examples by the largest difference (descending order)
    sorted_examples = sorted(examples, key=lambda x: x["Difference"], reverse=True)

    if output_fp:
        pd.DataFrame(sorted_examples).to_csv(output_fp, index=False, encoding='utf-8')
    # Return the top-k examples with the greatest difference
    return pd.DataFrame(sorted_examples[:k])

# Function to compute average ROUGE-L scores for a prediction across multiple references
def compute_avg_rouge(predictions, references):
    metric = evaluate.load("rouge")
    avg_scores = []

    for prediction, refs in zip(predictions, references):
        scores = []
        for ref in refs:  # Iterate through all references for a single prediction
            # Compute ROUGE-L score for each reference
            result = metric.compute(predictions=[prediction], references=[ref], use_stemmer=True)
            scores.append(result["rougeL"])
        # Compute the average ROUGE-L across all references
        avg_scores.append(sum(scores) / len(scores))
    return avg_scores

# Function to compute average BERTScore F1 for a prediction across multiple references
def compute_avg_bertscore(predictions, references, lang="en"):
    metric = evaluate.load("bertscore")
    avg_scores = []

    for prediction, refs in zip(predictions, references):
        scores = []
        for ref in refs:  # Iterate through all references for a single prediction
            # Compute BERTScore F1 for each reference
            result = metric.compute(predictions=[prediction], references=[ref], lang=lang)
            scores.append(result["f1"][0])  # Extract the F1 score
        # Compute the average BERTScore across all references
        avg_scores.append(sum(scores) / len(scores))
    return avg_scores

# Function to calculate ROUGE and BERTScore and find examples with the largest difference
def calculate_rouge_bertscore_multireferences(predictions, summary1_list, summary2_list, summary3_list, lang="en", k=10, output_fp=None):
    # Combine the three reference lists into a list of lists (one list of 3 references per prediction)
    references = [[ref1, ref2, ref3] for ref1, ref2, ref3 in zip(summary1_list, summary2_list, summary3_list)]

    # Compute the average ROUGE scores for all examples
    avg_rouge_scores = compute_avg_rouge(predictions, references)
    # Compute the average BERTScore F1 for all examples
    avg_bert_scores = compute_avg_bertscore(predictions, references, lang=lang)

    # Prepare a list to hold all examples with their scores
    examples = []
    for i, (prediction, refs, avg_rouge, avg_bert) in enumerate(zip(predictions, references, avg_rouge_scores, avg_bert_scores)):
        difference = avg_bert - avg_rouge  # Calculate the difference
        examples.append({
            "id": i,  # Keep the original ID
            "references": refs,
            "prediction": prediction,
            "Average_ROUGE": round(avg_rouge, 4),
            "Average_BERTScore": round(avg_bert, 4),
            "Difference": round(difference, 4),
        })

    # Sort examples by the largest difference (descending order)
    sorted_examples = sorted(examples, key=lambda x: x["Difference"], reverse=True)

    if output_fp:
        pd.DataFrame(sorted_examples).to_csv(output_fp, index=False, encoding='utf-8')

    # Return the top-k examples with the greatest difference
    return pd.DataFrame(sorted_examples[:k])

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    top_k = args.top_k

    if args.dataset == "dialogsum":
        test_file = args.test_file or "../data/dialogsum/dialogsum.test.jsonl"
        dialogue_list, summary1_list, summary2_list, summary3_list, ids = read_jsonlines_file(test_file)
        predictions = get_predictions(args.prompt_type, args.dataset, args.model)
        output_fp = os.path.join(args.output_dir, "sorted_examples_multireferences.csv")
        top_k_examples = calculate_rouge_bertscore_multireferences(
            predictions, summary1_list, summary2_list, summary3_list, lang="en", k=top_k, output_fp=output_fp
        )
    elif args.dataset == "decoda":
        test_file = args.test_file or "../data/decoda/test.json"
        dialogue_list, synopsis_list, ids = decoda_read_jsonlines_file(test_file)
        OneStep = ["Baseline", "Guideline_Original", "Guideline_Original_Annotator"]
        TwoStep = ["Guideline_Original_ToBaseline", "Guideline_Original_Annotator_ToBaseline"]
        if args.prompt_type in OneStep:
            df_pred = read_csv_to_df(f"../decoda/output/decoda_{args.model}_{args.prompt_type}.csv")
        elif args.prompt_type in TwoStep:
            df_pred = read_csv_to_df(f"../decoda/twoSteps/decoda_{args.model}_{args.prompt_type}.csv")
        else:
            raise ValueError("Prompt type does not exist.")
        predictions_212 = combine_ref_pred(ids, synopsis_list, df_pred)
        output_fp = os.path.join(args.output_dir, "sorted_examples.csv")
        top_k_examples = calculate_rouge_bertscore(
            predictions_212, synopsis_list, lang="fr", k=top_k, output_fp=output_fp
        )
    else:
        raise ValueError("Dataset does not exist.")

    print(top_k_examples)
    print(f"Saved sorted examples to {output_fp}")
