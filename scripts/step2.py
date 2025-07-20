"""
Serve as an intermediate step: 
Take the predictions from "Guideline_Original_Annotator", then skip the "Baseline" prompt which limits the length of output tokens to generate new summaries.
"""

import sys
import os
import csv
import random
import argparse
import time
import openai 
from openai import OpenAI
from src import read_csv_to_df
from openapi_summarization import predict_summary_dialogsum, predict_summary_decoda, dialog_dict_to_summ_dict, completion_with_backoff, get_openai_client

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dialogsum', help="The name of the dataset: dialogsum, decoda")
parser.add_argument('--prompt_type', default='Baseline', help="Different prompts: Guideline_Original, Guideline_AddEmo, Baseline")
parser.add_argument('--model_name', default='gpt-4', help="models")
parser.add_argument('--input_file_prompt', default='Guideline_Original_Annotator', help="HGR")
parser.add_argument('--api_key', default='', help="YOUR_API_KEY")
parser.add_argument('--sample', type=int, default=None, help="Number of random samples to process (default: all)")

if __name__ == '__main__':
	args = parser.parse_args()
	dataset = args.dataset
	prompt_type = args.prompt_type
	model_name = args.model_name
	input_file_prompt = args.input_file_prompt
	api_key = args.api_key

	if not api_key:
		api_key = os.environ.get("OPENAI_API_KEY")
		if not api_key:
			raise ValueError("API key not provided. Set --api_key or OPENAI_API_KEY environment variable.")
	client = OpenAI(api_key=api_key)

	output_dir = f"./{dataset}/twoSteps/"
	os.makedirs(output_dir, exist_ok=True)

	if dataset == "decoda":
		input_file = f"./decoda/output/decoda_{model_name}_{input_file_prompt}.csv"
	elif dataset == "dialogsum":
		input_file = f"./dialogsum/output/dialogsum_{model_name}_{input_file_prompt}.csv"
	else:
		raise ValueError(f"Unknown dataset: {dataset}")

	if not os.path.isfile(input_file):
		raise FileNotFoundError(f"Input file not found: {input_file}")

	df_pred = read_csv_to_df(input_file)
	dialog_dict_intermediate = dict(df_pred.values)

	# Sampling if requested
	if args.sample:
		random.seed(4)
		dialog_items = list(dialog_dict_intermediate.items())
		dialog_items = random.sample(dialog_items, min(args.sample, len(dialog_items)))
		dialog_dict_intermediate = dict(dialog_items)

	predicted_summaries = dialog_dict_to_summ_dict(
		client, dialog_dict_intermediate, prompt_type, model_name, dataset
	)

	output_file = os.path.join(
		output_dir, f"{dataset}_{model_name}_{input_file_prompt}_To{prompt_type}.csv"
	)

	with open(output_file, 'w', newline='', encoding='utf-8') as file:
		writer = csv.writer(file)
		writer.writerow(['TestID', 'Summary'])
		for ID, summary in predicted_summaries.items():
			writer.writerow([ID, summary])

	print(f'Done, {output_file} saved.')
