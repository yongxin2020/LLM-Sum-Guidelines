import json
import re

jsonl_file = "../results/human_annotations_decoda/decoda_erro_ana_20.jsonl"
mapping_file = "../results/human_annotations_decoda/targets_mapper_FR.json"
output_file = "./data/decoda_eval_samples.json"

# Function to clean HTML tags and replace <br> with new lines
def clean_text(text):
    text = re.sub(r"</?strong>", "", text)  # Remove <strong> tags
    text = text.replace("<br>", "\n").strip()  # Replace <br> with newline
    return text

# Load the system ID mappings
with open(mapping_file, "r", encoding="utf-8") as f:
    system_mappings = json.load(f)

# Process the JSONL file and extract relevant info
output_data = []

with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)  # Read JSONL line
        doc_id = entry["id"]  # Unique document ID
        texts = entry["text"]

        # Extract dialogue and clean it
        dialogue_v1 = clean_text(texts[0])
        dialogue = dialogue_v1.split("Dialogue: ")[1]

        # Collect system outputs and references
        predictions = texts[1:]
        reference_output = None  # Store reference separately

        # First pass: Identify the reference text
        for text in predictions:
            match = re.match(r"<strong>([A-Z]): <\/strong>(.*)", text.strip())
            if match:
                system_label, system_output = match.groups()
                system_output = clean_text(system_output)

                # Check if this is the reference
                if system_label in system_mappings and doc_id in system_mappings[system_label]:
                    system_id = system_mappings[system_label][doc_id]
                    if system_id == "reference":
                        reference_output = system_output  # Store the reference text

        # Second pass: Process all system outputs
        for text in predictions:
            match = re.match(r"<strong>([A-Z]): <\/strong>(.*)", text.strip())
            if match:
                system_label, system_output = match.groups()
                system_output = clean_text(system_output)

                # Get the system ID
                if system_label in system_mappings and doc_id in system_mappings[system_label]:
                    system_id = system_mappings[system_label][doc_id]
                else:
                    continue  # Skip if no valid system_id is found

                # Ensure reference is never empty
                final_reference = reference_output if reference_output else system_output

                # Append structured data
                output_data.append({
                    "doc_id": doc_id,  # Ensure each entry has the correct doc_id
                    "system_id": system_id,
                    "dialogue": dialogue,
                    "reference": final_reference,  # Ensure reference is always filled
                    "system_output": system_output
                })

# Save the output JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"Processed {len(output_data)} entries. Saved to {output_file}")