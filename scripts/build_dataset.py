"""
Preprocess the DECODA-EN test set for LLM experiments.

This script splits the DECODA dataset into train/dev/test sets and pairs each dialogue with its synopses,
removing tags and speaker labels. The processed data is saved as JSON files.
"""

import glob
import os
import json
import random
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../../Data/CCCS-Decoda-FR-EN/CCCS-Decoda-FR-EN-test_2015-04-10/EN',
                    help="Directory containing DECODA EN data (manual+auto translations)")
parser.add_argument('--output_dir', default='./build_dataset/test_EN',
                    help="Directory to save processed data")


def processText(all_text_files, all_synopsis_files, output_dir):
    for text_file in all_text_files:
        text_file_name = os.path.splitext(os.path.basename(text_file))[0]
        with open(text_file, 'r', encoding='latin-1') as ft:
            dialines = ft.readlines()
            DIALOGUE = ''.join(dialines)
            # 存在两种情况的标签：第一种在说话者标识（如A：）之后、话语开头：第二种在在说话者标识之前，上一句话结束\n换行符之前
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


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), f"Couldn't find the dataset at {args.data_dir}"

    train_data_dir = os.path.join(args.data_dir, 'train_EN')
    test_data_dir = os.path.join(args.data_dir, 'test_EN')

    test_text_filenames = glob.glob(os.path.join(test_data_dir, "text/*.txt"))
    test_synopsis_filenames = glob.glob(os.path.join(test_data_dir, "synopsis/FR_*.txt"))
    original_train_text_files = glob.glob(os.path.join(train_data_dir, "text/*.txt"))

    # Split train into 90% train, 10% dev (reproducible)
    random.seed(230)
    original_train_text_files.sort()
    random.shuffle(original_train_text_files)
    split = int(0.9 * len(original_train_text_files))
    train_filenames = original_train_text_files[:split]
    dev_filenames = original_train_text_files[split:]
    test_filenames = test_text_filenames

    filenames = {'train': train_filenames, 'valid': dev_filenames, 'test': test_filenames}
    original_train_synopsis_files = glob.glob(os.path.join(train_data_dir, "synopsis/FR_*.txt"))
    all_synopsis_files = original_train_synopsis_files + test_synopsis_filenames

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ['valid', 'train', 'test']:
        output_dir_split = os.path.join(args.output_dir, f'{split}_JSON_files')
        os.makedirs(output_dir_split, exist_ok=True)
        print(f"Processing {split} data, saving to {output_dir_split}")
        processText(filenames[split], all_synopsis_files, output_dir_split)

    print("Done building dataset")


