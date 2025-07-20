## Results obtained using 'evaluate.load' instead, from the new library 🤗

DATASETS=(decoda
dialogsum)

PROMPTS=(Baseline
Guideline_Original
Guideline_Original_Annotator
Guideline_Original_ToBaseline
Guideline_Original_Annotator_ToBaseline)

MODELS=(gpt-3.5
gpt-4
gpt-4o
)

PYTHON_PATH=YOUR_PYTHON_PATH_HERE

for dataset in "${DATASETS[@]}"; do
	for prompt in "${PROMPTS[@]}"; do
		for model in "${MODELS[@]}"; do
	    $PYTHON_PATH ./compute_metrics.py \
	      --dataset $dataset \
	      --prompt_type $prompt \
	      --model $model 
	   done
  done
done

