"""
For standard evaluation, no file paths (test_file, pred_file) are needed
"""

import sys
import os
import argparse
import json
import numpy as np
import jsonlines
from datasets import load_dataset, load_metric

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='decoda', help="The name of the dataset: dialogsum, decoda")
parser.add_argument('--prompt_type', default='Baseline', help="OneStep: Baseline, Guideline_Original, Guideline_Original_Annotator; TwoStep: Guideline_Original_ToBaseline, Guideline_Original_Annotator_ToBaseline")
parser.add_argument('--model', default='gpt-3.5', help="gpt-3.5, gpt-4, gpt-4o")
parser.add_argument('--test_file', default=None, help="Custom test file (only needed for non-standard evaluation)")
parser.add_argument('--pred_file', default=None, help="Custom prediction file (only needed for non-standard evaluation)")

def read_csv_to_df(file):
	import pandas as pd
	df = pd.read_csv(file, encoding='utf-8')
	return df

def read_jsonlines_file(file_name):
	dialogue_list = []
	summary1_list = []
	summary2_list = []
	summary3_list = []
	ids = []

	with jsonlines.open(file_name) as reader:
		for obj in reader:
			dialogue = obj["dialogue"]
			dialogue_list.append(dialogue)
			summary1 = obj["summary1"]
			summary1_list.append(summary1)
			summary2 = obj["summary2"]
			summary2_list.append(summary2)
			summary3 = obj["summary3"]
			summary3_list.append(summary3)
			id = obj["fname"]
			ids.append(id)
	# print(ids[:101])
	return dialogue_list, summary1_list, summary2_list, summary3_list, ids

def decoda_read_jsonlines_file(file_name):
	dialogue_list = []
	synopsis_list = []
	ids = []

	with jsonlines.open(file_name) as reader:
		for obj in reader:
			dialogue = obj["dialogue"]
			dialogue_list.append(dialogue)
			synopsis = obj["synopsis"]
			synopsis_list.append(synopsis)
			id = obj["id"]
			ids.append(id)
	# print(ids[:101])
	print("ids", len(set(ids))) # 100
	print("synopsis", len(set(synopsis_list))) # 212
	return dialogue_list, synopsis_list, ids

def combine_ref_pred(ids, synopsis_list, df_pred):
	"""
	Match predictions to reference IDs and return a list of predictions in reference order.
	"""
	import pandas as pd
	predictions = []
	df_ref = pd.DataFrame(
		{'ids': ids,
		 'synopsis': synopsis_list
		})
	for i in range(len(df_ref)):
		ID = df_ref['ids'][i]  # Access the 'ids' column
		# .values is used to convert the result to a NumPy array, and we check if there are any values before appending.
		prediction = df_pred.loc[df_pred['TestID'] == ID]["Summary"].values
		if len(prediction) > 0:
			predictions.append(prediction[0])  # Append the first matching prediction
	# df_new = df_ref.loc["Prediction"]=predictions
	return predictions

def compute_rouge(predictions_list, references_list):
	import datasets
	import evaluate
	metric = evaluate.load("rouge")
	# https://huggingface.co/spaces/evaluate-metric/rouge
	# This metrics is a wrapper around the Google Research reimplementation of ROUGE
	result = metric.compute(predictions=predictions_list, references=references_list, use_stemmer=True) #summary1_list, summary2_list, summary3_list
	result = {key: value * 100 for key, value in result.items()} # fmeasure
	result = {k: round(v, 2) for k, v in result.items()}
	return result

def get_predictions(prompt_type, dataset, model, pred_file=None):
	# Define prompt type directories
	OneStep = ["Baseline", "Guideline_Original", "Guideline_Original_Annotator"]
	TwoStep = ["Guideline_Original_ToBaseline", "Guideline_Original_Annotator_ToBaseline"]

	input_dir = f"../{dataset}/output" if prompt_type in OneStep else f"../{dataset}/twoSteps" if prompt_type in TwoStep else None

	# Dataset-specific file paths for "bart-based"
	bart_file_paths = {
		"decoda": "../decoda/sota_barthez_generated_predictions.txt",
		"dialogsum": "../dialogsum/bart_large_generated_summaries.txt"
	}

	# Use pred_file argument if provided
	if pred_file:
		file_path = pred_file
	elif model in ["gpt-3.5", "gpt-4", "gpt-4o"]:
		file_path = os.path.join(input_dir, f"{dataset}_{model}_{prompt_type}.csv")
	elif model == "bart-based":
		file_path = bart_file_paths.get(dataset)
		if not file_path:
			raise ValueError(f"Unsupported dataset '{dataset}' for bart-based model.")
	else:
		raise ValueError("Model name does not exist.")

	# Read predictions
	if model == "bart-based":
		predictions = read_txt_to_list(file_path)
	else:
		df = read_csv_to_df(file_path)
		predictions = df["Summary"].tolist()

	# Apply dataset-specific modifications
	if dataset == "dialogsum":
		predictions = [p.replace("Person1", "#Person1#").replace("Person2", "#Person2#") for p in predictions]

	return predictions

def calculate_rouge_bertscore(predictions, references, lang="en"):
	from evaluate import load
	bertscore = load("bertscore")

	result_rouge = compute_rouge(predictions, references)
	print("ROUGE results", result_rouge)

	results = bertscore.compute(predictions=predictions, references=references, lang=lang) # change the "model_type" # , model_type="distilbert-base-uncased", default for others: bert-base-multilingual-cased
	print("results f1", round(sum(results["f1"])/len(predictions), 4))

def read_txt_to_list(file):
	with open(file) as f:
		content_list = [line for line in f]
	return content_list

if __name__ == '__main__':
	args = parser.parse_args()
	dataset = args.dataset
	prompt_type = args.prompt_type
	model = args.model
	test_file = args.test_file
	pred_file = args.pred_file

	OneStep = ["Baseline", "Guideline_Original", "Guideline_Original_Annotator"]
	TwoStep = ["Guideline_Original_ToBaseline", "Guideline_Original_Annotator_ToBaseline"]

	if dataset == "dialogsum":
		if not test_file:
			test_file = "../data/dialogsum/dialogsum.test.jsonl"
		dialogue_list, summary1_list, summary2_list, summary3_list, ids = read_jsonlines_file(test_file)
		if model == "bart-based":
			bart_large_path = pred_file if pred_file else "../dialogsum/bart_large_generated_summaries.txt"
			predictions = read_txt_to_list(bart_large_path)
			print("bart_large_predictions", model)
			# 1500 predictions; each sample's prediction repeats three times.
			# zip(...) groups summaries by sample.
			# The list comprehension flattens all summaries into one list.
			# references: [summary1_1, summary2_1, summary3_1, summary1_2, ...]			references = [summary for summaries in zip(summary1_list, summary2_list, summary3_list) for summary in summaries]
			calculate_rouge_bertscore(predictions, references, lang="en")
		else:
			predictions = get_predictions(prompt_type, dataset, model, pred_file)
			predictions_3times = predictions * 3
			references = summary1_list + summary2_list + summary3_list
			calculate_rouge_bertscore(predictions_3times, references, lang="en")

		print("Example predictions:", predictions[5:10])
		print("Example Summary1:", summary1_list[5:10])
		print("Example Summary2:", summary2_list[5:10])
		print("Example Summary3:", summary3_list[5:10])

	elif dataset == "decoda":
		if not test_file:
			test_file = "../data/decoda/test.json"
		dialogue_list, synopsis_list, ids = decoda_read_jsonlines_file(test_file)

		base_path = "../decoda/output" if prompt_type in OneStep else "../decoda/twoSteps" if prompt_type in TwoStep else None

		if model in ["gpt-3.5", "gpt-4", "gpt-4o"]:
			if test_file or pred_file:
				print("Warning: test_file and pred_file are ignored for GPT model evaluation.")
		if model in ["gpt-3.5", "gpt-4", "gpt-4o"]:
			if pred_file:
				pred_path = pred_file
			else:
				file_name = f"decoda_{model}_{prompt_type}.csv"
				pred_path = os.path.join(base_path, file_name)
			if not os.path.isfile(pred_path):
				raise FileNotFoundError(f"Prediction file not found: {pred_path}")
			df_pred = read_csv_to_df(pred_path)
			print(f"Evaluating: {prompt_type}, {model}")
			predictions_212 = combine_ref_pred(ids, synopsis_list, df_pred)
		elif model == "bart-based":
			# Use pred_file if provided, otherwise use default for each dataset
			barthez_path = pred_file if pred_file else "../decoda/sota_barthez_generated_predictions.txt"
			if not os.path.isfile(barthez_path):
				raise FileNotFoundError(f"BARThez prediction file not found: {barthez_path}")
			predictions_212 = read_txt_to_list(barthez_path)
			print("BARThez_predictions", model)
		else:
			raise ValueError(f"Unknown model: {model}")

		print("Example predictions:", predictions_212[5:10])
		print("Example references:", synopsis_list[5:10])
		calculate_rouge_bertscore(predictions_212, synopsis_list, lang="fr")