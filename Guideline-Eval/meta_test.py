import json
from collections import defaultdict
import statistics
import csv
import argparse
import re

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--json_fp', type=str, default='./results/gpt-4/gpt4_faithfulness.json', help="JSON file")
    argparser.add_argument('--save_fp', type=str, default='./results/gpt-4/gpt4_faithfulness.csv', help="Save CSV file")
    args = argparser.parse_args()

    # Load the JSON file
    with open(args.json_fp, 'r') as f:
        data = json.load(f)

    # Dictionary to store scores for each system
    system_scores = defaultdict(list)
    # Dictionary to store scores for each doc_id and system_id
    doc_system_scores = defaultdict(dict)

    # Extract scores from the "response" field
    for entry in data:
        doc_id = entry['doc_id']
        system_id = entry['system_id']
        response = entry['response']

        # Extract the numeric score from the response
        # Assuming the score is in the format: "- Faithfulness : \n5"
        #         "response": "5"
        try:
            # Extract the first numeric value (integer or float) from the response string.
            match = re.search(r"\d+(\.\d+)?", response)
            
            if match:
                score = float(match.group())  # Convert the matched number to float
                if 1 <= score <= 5:
                    system_scores[system_id].append(score)
                    doc_system_scores[doc_id][system_id] = score
            else:
                raise ValueError("No valid number found")  # Explicitly raise an error if no number is found

        except (AttributeError, ValueError, IndexError) as e:
            print(f"Could not extract score from response: {response}. Error: {e}")

    # Calculate mean and standard deviation for each system
    results = {}
    for system_id, scores in system_scores.items():
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0  # Avoid division by zero for single score
        nums = len(scores)
        results[system_id] = {'mean': mean, 'std': std, 'nums':nums}

    # Print the results
    for system_id, metrics in results.items():
        print(f"System ID: {system_id}, Mean: {metrics['mean']:.2f}, Std: {metrics['std']:.2f}, Valid response number: {metrics['nums']}")

    # Get all unique system_ids
    system_ids = set()
    for scores in doc_system_scores.values():
        system_ids.update(scores.keys())
    system_ids = sorted(system_ids)  # Sort for consistent column order

    # Write to CSV
    with open(args.save_fp, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['doc_id'] + system_ids)
        writer.writeheader()

        for doc_id, scores in doc_system_scores.items():
            row = {'doc_id': doc_id}
            row.update(scores)  # Add scores for each system_id
            writer.writerow(row)

    print(f"CSV file '{args.save_fp}.csv' created successfully.")
