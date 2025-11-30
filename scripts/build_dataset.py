"""
Preprocess the DECODA-FR dataset for LLM experiments.

This script processes the original DECODA dataset (French), splitting it into train, dev, and test sets.
It pairs each dialogue with its corresponding synopses, removes tags and speaker labels, and saves the result as JSON files.

Input Data Structure:
The script expects separate directories for training and testing data (provided via --train_dir and --test_dir).
Within each directory, it looks for data in 'manual' and 'auto' subfolders, expecting the following structure:
    - manual/text/*.txt
    - manual/synopsis/FR_*.txt
    - auto/text/*.txt
    - auto/synopsis/FR_*.txt

Output:
The processed data is saved to the specified output directory (default: ../data/decoda/processed).
It creates 'train_JSON_files', 'valid_JSON_files', and 'test_JSON_files'.

The final file used in the experiments of this repo should be saved in ../data/decoda/test.json.
This file contains 212 lines, corresponding to the 100 samples in the test set.
Since each sample can have more than one synopsis (up to 5), we pair up each synopsis with the corresponding dialogue.
This results in multiple entries for the same dialogue if it has multiple synopses, creating the final test.json structure.
"""

import glob
import os
import json
import random
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', required=True, help="Directory containing original DECODA FR train data")
parser.add_argument('--test_dir', required=True, help="Directory containing original DECODA FR test data")
parser.add_argument('--output_dir', default='../data/decoda/processed',
                    help="Directory to save processed data")


def processText(all_text_files, all_synopsis_files, output_dir):
    for text_file in all_text_files:
        text_file_name = os.path.splitext(os.path.basename(text_file))[0]
        with open(text_file, 'r', encoding='latin-1') as ft:
            dialines = ft.readlines()
            DIALOGUE = ''.join(dialines)
            # There are two types of tags: 1. After the speaker label (e.g., A:) at the start of the utterance.
            # 2. Before the speaker label, before the newline character \n of the previous sentence.
            # Remove tags (e.g., <noise b/>) and placeholder for phone beep
            DIALOGUE = re.sub(r"(\s)?<[^>]*>", '', DIALOGUE)
            DIALOGUE = DIALOGUE.replace(" NNAAMMEE", "")

            # Remove speaker labels (e.g., "A:\n", "B:\n", etc.)
            DIALOGUE = re.sub(r"[A-E]:\n", '', DIALOGUE)

            # Remove duplicate or concatenated speaker labels (e.g., "A: B:", "B: C:", etc.)
            DIALOGUE = re.sub(r"([A-E]: ){1,}[A-E]:", lambda m: m.group(0)[-2:], DIALOGUE)

            # Remove repeated speaker labels (e.g., "A: A:", "B: B:", etc.)
            DIALOGUE = re.sub(r"([A-E]): \1:", r"\1:", DIALOGUE)

            ## Optionally, remove any remaining single speaker labels at the start of a line (for No_Spk version)
            # DIALOGUE = re.sub(r"^[A-E]:", '', DIALOGUE, flags=re.MULTILINE)

            ID = text_file_name

        # Pair each dialogue with its synopses
        for synopsis_file in all_synopsis_files:
            synopsis_file_name = os.path.splitext(os.path.basename(synopsis_file))[0]
            new_synopsis_file_name = synopsis_file_name.replace('_syn', '')
            if text_file_name != new_synopsis_file_name:
                continue
            with open(synopsis_file, encoding='utf-8') as fh:
                lines = fh.readlines()
                for idx, line in enumerate(lines[:5]):
                    # Each line: "AnnotatorID SynopsisText"
                    parts = line.strip().split(None, 2)
                    if len(parts) < 3:
                        continue
                    synopsis = parts[2]
                    DICT = {'id': ID, 'synopsis': synopsis, 'dialogue': DIALOGUE}
                    newfile = os.path.join(output_dir, f'{text_file_name}_synopsis_{idx+1}.json')
                    with open(newfile, 'w', encoding='utf-8') as output:
                        json.dump(DICT, output, indent=4, ensure_ascii=False)
                        output.write('\n')


def get_files_from_dir(data_dir):
    text_files = []
    synopsis_files = []
    
    # Manual
    manual_text = glob.glob(os.path.join(data_dir, 'manual', 'text', '*.txt'))
    manual_synopsis = glob.glob(os.path.join(data_dir, 'manual', 'synopsis', 'FR_*.txt'))
    text_files.extend(manual_text)
    synopsis_files.extend(manual_synopsis)
    
    # Auto
    auto_text = glob.glob(os.path.join(data_dir, 'auto', 'text', '*.txt'))
    auto_synopsis = glob.glob(os.path.join(data_dir, 'auto', 'synopsis', 'FR_*.txt'))
    text_files.extend(auto_text)
    synopsis_files.extend(auto_synopsis)
    
    return text_files, synopsis_files


if __name__ == '__main__':
    args = parser.parse_args()
    
    if not os.path.isdir(args.train_dir):
        raise ValueError(f"Couldn't find the train dataset at {args.train_dir}")
    if not os.path.isdir(args.test_dir):
        raise ValueError(f"Couldn't find the test dataset at {args.test_dir}")

    # Get files from original directories
    train_text_files, train_synopsis_files = get_files_from_dir(args.train_dir)
    test_text_files, test_synopsis_files = get_files_from_dir(args.test_dir)

    # Split train into 90% train, 10% dev (reproducible)
    random.seed(230)
    train_text_files.sort()
    random.shuffle(train_text_files)
    
    split = int(0.9 * len(train_text_files))
    train_filenames = train_text_files[:split]
    dev_filenames = train_text_files[split:]
    test_filenames = test_text_files

    filenames = {'train': train_filenames, 'valid': dev_filenames, 'test': test_filenames}
    all_synopsis_files = train_synopsis_files + test_synopsis_files

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ['valid', 'train', 'test']:
        output_dir_split = os.path.join(args.output_dir, f'{split}_JSON_files')
        os.makedirs(output_dir_split, exist_ok=True)
        print(f"Processing {split} data, saving to {output_dir_split}")
        processText(filenames[split], all_synopsis_files, output_dir_split)

    print("Done building dataset")


