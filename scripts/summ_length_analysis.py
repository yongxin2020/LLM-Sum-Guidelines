import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import nltk
import jsonlines

def read_jsonlines_to_dict(test_file):
    import jsonlines
    dialog_dict = {}

    # sys.argv is automatically a list of strings representing the arguments (as separated by spaces) on the command-line
    with jsonlines.open(test_file) as reader: #sys.argv[1]
        for row in reader:
            dialog_dict[row['fname']] = row['dialogue']

    print("Number of dialogues:", len(dialog_dict))
    return dialog_dict

def shorten_dialog(dialog_str, max_token_length):
    import nltk
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')

    # Split the dialogue into utterances (lines)
    print(len(dialog_str))
    utterances = dialog_str.split('\n')

    # Initialize variables
    shortened_utterances = []
    current_length = 0

    for utterance in utterances:
        words = word_tokenize(utterance)
        # Calculate the length of the current utterance (including spaces)
        utterance_length = sum(len(word) + 1 for word in words)  # Add 1 for spaces
        # Check if adding the current utterance exceeds the maximum length
        if current_length + utterance_length > max_token_length:
            break  # Stop if the limit is reached
        # Add the utterance to the shortened dialogue
        shortened_utterances.append(utterance)
        # Update the current length
        current_length += utterance_length

    # Join the shortened utterances to create the final shortened dialogue
    shortened_dialog_str = '\n'.join(shortened_utterances)
    print(len(shortened_dialog_str))

    return shortened_dialog_str

def read_csv_to_df(file):
    return pd.read_csv(file, encoding='utf-8')

def decoda_read_jsonlines_file(file_name):
    dialogue_list, synopsis_list, ids = [], [], []
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            dialogue_list.append(obj["dialogue"])
            synopsis_list.append(obj["synopsis"])
            ids.append(obj["id"])
    print("ids", len(set(ids)))
    print("synopsis", len(set(synopsis_list)))
    return dialogue_list, synopsis_list, ids

# # predictions212: 100 dialogues but actually 212 synopses, to compare the output length, we need to make our predictions also 212
def combine_ref_pred(ids, synopsis_list, df_pred): # function from compute_metrics.py
    predictions = []
    df_ref = pd.DataFrame({'ids': ids, 'synopsis': synopsis_list})
    for i in range(len(df_ref)):
        ID = df_ref['ids'][i]  # Access the 'ids' column
        # .values is used to convert the result to a NumPy array, and we check if there are any values before appending.
        prediction = df_pred.loc[df_pred['TestID'] == ID]["Summary"].values
        if len(prediction) > 0:
            predictions.append(prediction[0])  # Append the first matching prediction
    return predictions

def get_predictions(dataset, prompt_type, model_name):
    """Get predictions for a given dataset/model/prompt."""
    if prompt_type in ["Baseline", "Guideline_Original", "Guideline_Original_Annotator"]:
        step_dir = "output"
    elif prompt_type in ["Guideline_Original_ToBaseline", "Guideline_Original_Annotator_ToBaseline"]:
        step_dir = "twoSteps"
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")
    input_dir = f"../{dataset}/{step_dir}/"
    file_path = os.path.join(input_dir, f"{dataset}_{model_name}_{prompt_type}.csv")
    if not os.path.isfile(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    df = read_csv_to_df(file_path)
    if dataset == "dialogsum":
        predictions = [(pre.replace("Person1", "#Person1#")).replace("Person2", "#Person2#") for pre in df.Summary]
    elif dataset == "decoda":
        test_file = "../data/decoda/test.json"
        _, _, ids = decoda_read_jsonlines_file(test_file)
        predictions = combine_ref_pred(ids, df)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return predictions

def get_summ_length_from_prompt(dataset, prompt_type, model_name):
    predictions = get_predictions(dataset, prompt_type, model_name)
    if predictions is None:
        return None
    nltk.download('punkt', quiet=True)
    summ_length = []
    for summary in predictions:
        pre_modified = (summary.replace("#Person1#", "Person1")).replace("#Person2#", "Person2")
        summary_len = len(nltk.word_tokenize(pre_modified))
        summ_length.append(summary_len)
    return summ_length

def collect_summary_lengths(dataset, model_names, prompt_types):
    """Collect summary lengths for all model/prompt combinations."""
    results = {}
    max_len = 0
    for model_name in model_names:
        for prompt_type in prompt_types:
            key = f"{model_name}-{prompt_type}"
            summ_lens = get_summ_length_from_prompt(dataset, prompt_type, model_name)
            if summ_lens is not None:
                results[key] = summ_lens
                max_len = max(max_len, len(summ_lens))
            else:
                print(f"Skipping {key} due to missing predictions.")
    # Pad shorter lists with np.nan for DataFrame alignment
    for k in results:
        if len(results[k]) < max_len:
            results[k] += [np.nan] * (max_len - len(results[k]))
    return pd.DataFrame(results)

def plot_summary_lengths(data, dataset, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    data.boxplot(grid=False)
    plt.ylabel('Summary Length')
    plt.xlabel('Model-Prompt')
    plt.title(f'Summary Lengths for {dataset}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_summary_lengths.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Figure saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='decoda', help="Dataset: dialogsum or decoda")
    parser.add_argument('--model_names', nargs='+', default=['gpt-4o'], help="List of model names")
    parser.add_argument('--prompt_types', nargs='+', default=['Baseline', 'Guideline_Original_Annotator', 'Guideline_Original_Annotator_ToBaseline'], help="List of prompt types")
    parser.add_argument('--plot', action='store_true', help="Whether to plot summary length boxplots")
    parser.add_argument('--out_dir', default='../results/Fig/LA', help="Output directory for figures")
    args = parser.parse_args()

    data = collect_summary_lengths(args.dataset, args.model_names, args.prompt_types)
    print(data.describe())

    if args.plot:
        plot_summary_lengths(data, args.dataset, out_dir=args.out_dir)

