import jsonlines
import sys
import os
import glob
import csv
import random
import argparse
import time
import openai
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from src import read_jsonlines_to_dict, shorten_dialog

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='decoda', help="The name of the dataset: dialogsum, decoda")
parser.add_argument('--input_dir', default='./data/', help="Where the dialogue summarization corpus is stored")
parser.add_argument('--output_dir', default=None, help="Directory to save model outputs (default: ./<dataset>/output/)")
parser.add_argument('--model_name', default='gpt-3.5-turbo', help="gpt-3.5-turbo, gpt-4, gpt-4o")
parser.add_argument('--prompt_type', default='Baseline', help="Different types of prompts: Guideline_Original, Guideline_Original_Annotator, Baseline, etc")
parser.add_argument('--api_key', default='', help="YOUR_API_KEY")
parser.add_argument('--sample', type=int, default=None, help="Number of random samples to process (default: all)")

# Calculate the delay based on your rate limit
rate_limit_per_minute = 20
delay = 60.0 / rate_limit_per_minute

def get_openai_client(api_key):
	"""Initialize OpenAI client with the provided API key."""
	return OpenAI(api_key=api_key)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(client, **kwargs):
	"""Retries API call with exponential backoff in case of failure."""
	return client.chat.completions.create(**kwargs)

def predict_summary_decoda(client, dialog_str, prompt_type, model_name):
	if model_name=="gpt-3.5-turbo":
		dialog_str = shorten_dialog(dialog_str, 10294) # 4096 # 11044: Baseline, 10294: Guideline_Original_Baseline
	else:
		pass

	if prompt_type == "Guideline_Original":
		response = completion_with_backoff(
		  # delay_in_seconds=delay,
		  client,
		  model=model_name,
		  messages=[
		{
		"role": "system",
		"content": "Rédiger un résumé axé sur la conversation sous la forme d'un synopsis exprimant à la fois le point de vue du client et celui de l'agent, et qui devrait idéalement faire état de ce qui suit : 1. Les questions principales de la conversation : dans les conversations des centres d'appel, les questions principales sont les problèmes pour lesquels le client a appelé ; leur identification constitue la base de la classification de l'appel dans plusieurs classes différentes de motivations d'appel. Le problème principal d'un appel doit être classé par ordre de priorité pour être inclus dans le résumé du même appel.  2. Les sous-problèmes de la conversation : lorsqu'un sous-problème apparaît dans la conversation, il peut être introduit par le client ou par les agents. 3. La résolution de l'appel : c'est-à-dire si le problème du client a été résolu au cours de cet appel (résolution au premier appel) ou non."}, # You are also asked to write a short (around 3 tokens) topic for each dialogue. # Le synopsis doit être d'environ 25 mots.
		{
			"role": "user", "content": dialog_str}
			],
			# max_tokens=64,
			temperature=0
		)

	elif prompt_type == "Guideline_Original_Annotator":    # it begins with "you are an annotator ..."
		response = completion_with_backoff(
		  # delay_in_seconds=delay,
		  client,
		  model=model_name,
		  messages=[
		{"role": "system",
		"content": "Vous êtes un annotateur et il vous est demandé de rédiger un résumé axé sur la conversation sous la forme d'un synopsis exprimant à la fois le point de vue du client et celui de l'agent, et qui devrait idéalement faire état de ce qui suit : 1. Les questions principales de la conversation : dans les conversations des centres d'appel, les questions principales sont les problèmes pour lesquels le client a appelé ; leur identification constitue la base de la classification de l'appel dans plusieurs classes différentes de motivations d'appel. Le problème principal d'un appel doit être classé par ordre de priorité pour être inclus dans le résumé du même appel.  2. Les sous-problèmes de la conversation : lorsqu'un sous-problème apparaît dans la conversation, il peut être introduit par le client ou par les agents. 3. La résolution de l'appel : c'est-à-dire si le problème du client a été résolu au cours de cet appel (résolution au premier appel) ou non."}, # You are also asked to write a short (around 3 tokens) topic for each dialogue. # Le synopsis doit être d'environ 25 mots.
		{"role": "user", "content": dialog_str}
			],
			# max_tokens=64,
			temperature=0
		)

	elif prompt_type == "Baseline":
	# response = openai.ChatCompletion.create(
		response = completion_with_backoff(
		  # delay_in_seconds=delay,
		  client,
		  model=model_name, #gpt-4, gpt-3.5-turbo
		  messages=[
		{"role": "system", "content": "Rédiger un résumé (synopsis) en 25 mots maximum."},
		{"role": "user", "content": dialog_str} # here should I put only the dialogue input or with specific instructions?
			],
			#max_tokens=100,
			temperature=0
		)

	text = response.choices[0].message.content
	return text

def predict_summary_dialogsum(client, dialog_str, prompt_type, model_name):
	if model_name=="gpt-3.5-turbo":
		dialog_str = shorten_dialog(dialog_str, 10294) # 4096 # 11044: Baseline, 10294: Guideline_Original_Baseline
	else:
		pass

	if prompt_type == "Guideline_Original":
		response = completion_with_backoff(
		  # delay_in_seconds=delay,
		  client,
		  model=model_name,
		  messages=[
		{"role": "system", "content": "Write a summary based on following criteria: the summary should (1) convey the most salient information of the dialogue and; (2) be brief (no longer than 20% of the conversation length) and; (3) preserve important named entities within the conversation and; (4) be written from an observer perspective and; (5) be written in formal language. In addition, you should pay extra attention to the following aspects: 1) Tense Consistency: take the moment that the conversation occurs as the present time, and choose a proper tense to describe events before and after the ongoing conversation. 2) Discourse Relation: If summarized events hold important discourse relations, particularly causal relation, you should preserve the relations if they are also in the summary. 3) Emotion: you should explicitly describe important emotions related to events in the summary. 4) Intent Identification: Rather than merely summarizing the consequences of dialogues, you should also describe speakers’ intents in summaries, if they can be clearly identified. In addition to the above, you should use person tags to refer to different speakers if real names cannot be detected from the conversation."}, # You are also asked to write a short (around 3 tokens) topic for each dialogue.
		{"role": "user", "content": dialog_str}
			],
			# max_tokens=64,
			temperature=0
		)

	elif prompt_type == "Guideline_Original_Annotator":  # Guideline_WithEmo
		response = completion_with_backoff(
		  # delay_in_seconds=delay,
		  client,
		  model=model_name,
		  messages=[
		{"role": "system", "content": "You are an annotator and are asked to write dialogue summaries based on following criteria: the summary should (1) convey the most salient information of the dialogue and; (2) be brief (no longer than 20% of the conversation length) and; (3) preserve important named entities within the conversation and; (4) be written from an observer perspective and; (5) be written in formal language. In addition, you should pay extra attention to the following aspects: 1) Tense Consistency: take the moment that the conversation occurs as the present time, and choose a proper tense to describe events before and after the ongoing conversation. 2) Discourse Relation: If summarized events hold important discourse relations, particularly causal relation, you should preserve the relations if they are also in the summary. 3) Emotion: you should explicitly describe important emotions related to events in the summary. 4) Intent Identification: Rather than merely summarizing the consequences of dialogues, you should also describe speakers’ intents in summaries, if they can be clearly identified. In addition to the above, you should use person tags to refer to different speakers if real names cannot be detected from the conversation."}, # You are also asked to write a short (around 3 tokens) topic for each dialogue.
		{"role": "user", "content": dialog_str}
			],
			# max_tokens=64,
			temperature=0
		)

	elif prompt_type == "Baseline":
		response = completion_with_backoff(
		  # delay_in_seconds=delay,
		  client,
		  model=model_name,
		  messages=[
		{"role": "system", "content": "Writing a summary in not more than 20 words"}, #Write a very short and concise summary of the following dialogue in not more than 20 words.
		{"role": "user", "content": dialog_str}
			],
			#max_tokens=100,
			temperature=0
		)
	# print(response['choices'])
	text = response.choices[0].message.content

	return text

def get_random_sample(split):
	# data = dataset[split]
	glob.glob(os.path.join(input_dir, f"{split}.json"))
	index = random.randint(0, len(data))
	return data[index]['dialogue'], data[index]['summary']

def split(output):
	topic = output.split("\n")[0].split(": ")[1]
	summary = output.split("\n")[2]
	return topic, summary

def dialog_dict_to_summ_dict(client, dialog_dict, prompt_type, model_name, dataset):
	predicted_summaries = {}
	for ID, dialog in dialog_dict.items():
		if dataset == "decoda":
			summary = predict_summary_decoda(client, dialog, prompt_type, model_name)
		elif dataset == "dialogsum":
			summary = predict_summary_dialogsum(client, dialog, prompt_type, model_name)
		predicted_summaries[ID] = summary
		# print(f"{ID} => {summary}") # for debug use
	return predicted_summaries

if __name__ == '__main__':
	args = parser.parse_args()
	dataset = args.dataset
	input_dir = args.input_dir
	model_name = args.model_name
	prompt_type = args.prompt_type
	api_key = args.api_key

	if not api_key:
		api_key = os.environ.get("OPENAI_API_KEY")
		if not api_key:
			raise ValueError("API key not provided. Set --api_key or OPENAI_API_KEY environment variable.")
	client = OpenAI(api_key=api_key)

	if args.output_dir:
		output_dir = args.output_dir
	else:
		output_dir = os.path.join('.', dataset, 'output')
	os.makedirs(output_dir, exist_ok=True)

	if dataset == "decoda":
		test_file = glob.glob(os.path.join(input_dir, "./decoda/test.json")) # spk version of DECODA data
		print(test_file)

		dialog_dict = {}

		with jsonlines.open(test_file[0]) as reader:
			for row in reader:
				dialog_dict[row['id']] = row['dialogue']

	elif dataset == "dialogsum":
		test_file = glob.glob(os.path.join(input_dir, "./dialogsum/dialogsum.test.jsonl"))
		dialog_dict = read_jsonlines_to_dict(test_file[0])

	if not test_file:
		raise FileNotFoundError(f"No test file found for dataset {dataset} in {input_dir}")

	print("Number of dialogues:", len(dialog_dict))

	random.seed(4)
	if args.sample:
		dialog_dict = dict(random.sample(dialog_dict.items(), args.sample))

	predicted_summaries = dialog_dict_to_summ_dict(client, dialog_dict, prompt_type, model_name, dataset)

	output_file = os.path.join(output_dir, f"{dataset}_{model_name}_{prompt_type}.csv")

	with open(output_file, 'w', encoding='utf-8') as file: # don't forget encoding='utf-8' for French characters
		writer = csv.writer(file)
		writer.writerow(['TestID', 'Summary'])

		for ID, summary in predicted_summaries.items():
			writer.writerow([ID, summary])

	print(f'Done, {output_file} saved.')
