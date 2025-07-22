# Code for paper "Can GPT models Follow Human Summarization Guidelines? A Study for Targeted Communication Goals" [https://arxiv.org/abs/2310.16810]

****
<span id='content'/>

## Content: 
1. [Introduction](#introduction)  
2. [Code Structure](#code_structure)  
3. [Experiments](#experiments)  
   - 3.1 [Data Preparation](#data_preparation)  
   - 3.2 [Prompt Design](#prompt_design)  
   - 3.3 [DialogSum Summarization](#ds_dialogsum)  
   - 3.4 [DECODA-FR Summarization](#ds_decoda_fr)  
4. [Results](#results)  
   - 4.1 [Quantitative Evaluation](#quantitative_evaluation)
   - 4.2 [Example Analysis](#example_analysis)
   - 4.3 [Human Evaluation](#human_evaluation)
   - 4.4 [Summary Length Analysis](#length_analysis)
5. [Citation](#citation)  

****

<span id='introduction'/>

## 1. Introduction <a href='#content'>[Back to Top]</a>
This repository accompanies the paper *"Can GPT models Follow Human Summarization Guidelines? A Study for Targeted Communication Goals"*, which investigates the ability of prompt-driven LLMs (e.g., ChatGPT, GPT-4) to adhere to human guidelines for dialogue summarization. Experiments were conducted on:  
- **DialogSum**: English social conversations  
- **DECODA-FR**: French call center interactions  

Key findings:  
- **GPT models** (ChatGPT, GPT-4, GPT-4o) outperform task-specific models and even reference summaries in human evaluation, likely due to longer, more comprehensive outputs.
- Despite lower **automatic metric scores** (ROUGE/BERTScore), GPT summaries are preferred by humans, highlighting the need for better-aligned metrics and continued human evaluation.
- **Guideline adherence**: GPT-4 better follows word limits, while the HGR→WL approach yields superior results over simple WordLimit.
- **Subjectivity in evaluation**: Human judges often favor GPT summaries over references due to stylistic differences.
- **Shortcomings**: GPT models occasionally miss rules (e.g., named entities in DialogSum or balanced perspectives in DECODA), though HGR intermediate steps improve adherence.

<span id='code_structure'/>

## 2. Code Structure <a href='#code_structure'>[Back to Top]</a>
```
├── decoda/ # DECODA-FR experiments
│ ├── output/ # One-step prompt outputs
│ ├── twoSteps/ # Two-step prompt outputs
│ └── sota_barthez_predictions.txt # BARTthez fine-tuned predictions
├── dialogsum/ # DialogSum experiments
│ ├── output/ # One-step prompt outputs
│ ├── twoSteps/ # Two-step prompt outputs
│ └── bart_large_summaries.txt # BART-large fine-tuned predictions
├── scripts/
│ ├── build_dataset.py # Preprocess DECODA raw data
│ ├── compute_metrics.py # ROUGE/BERTScore evaluation
│ ├── example_analysis.py # Data points of generated summaries (low ROUGE but high BERTScore)
│ ├── human_eval_save_annotated_json.py # Process human evaluation annotations and save evaluation samples with annotation scores
│ ├── human_eval_scores.py # Aggregate and display human evaluation scores
│ ├── openapi_summarization.py # GPT summarization experiments
│ ├── step2.py # Intermediate step for two-step prompting
│ └── summ_length_analysis.py # Length analysis & box plots
│ └── variance.py # Human-vs-model variance analysis
├── Guideline-Eval/ # LLM-based evaluation scripts and prompts
└── results/ # Evaluation outputs, figures, and human annotation data
```

### Setup
```bash
conda create --name <yourEnv> python=3.8
conda activate <yourEnv>
pip install -r requirements.txt
```

<span id='experiments'/>

## 3. Experiments <a href='#experiments'>[Back to Top]</a>

<span id='data_preparation'/>

### Data Preparation

#### Datasets
- *DECODA*: Download from [MultiLing 2015 -- CCCS data download](https://pageperso.lis-lab.fr/benoit.favre/cccs/) (test set requires author approval).
- *DialogSum*: Download from [GitHub](https://github.com/cylnlp/dialogsum#dialogsum-a-real-life-scenario-dialogue-summarization-dataset).

#### Preprocessing
- *DECODA*: Speaker turns are marked with `<Spk A>`, `<Spk B>`, etc. Noise labels (e.g., `<noise b/>`) are filtered.
```bash
python ./scripts/build_dataset.py
```

You can save the data downloaded (and preprocessed) in the [data](./data) repo.
- *DECODA*: Test file path `./data/decoda/test.json`
- *DialogSum*: Test data path `./data/dialogsum/dialogsum.test.jsonl`

<span id='prompt_design'/>

### Prompt Design
Prompts include:
1. **Baseline**: Word-length constraints.
2. **Guideline_Original**: Human summarization guidelines.
3. **Guideline_Original_Annotator**: Guidelines begin with "you are an annotator ...".

Refer to [OpenAI’s prompt guide](https://platform.openai.com/docs/guides/text-generation?lang=python#building-prompts) for design principles.

<span id='ds_dialogsum'/>

### DialogSum Summarization
```bash
python ./scripts/openapi_summarization.py \
    --dataset dialogsum  \
    --model_name gpt-4o \
    --input_dir ./data/ \
    --prompt_type Baseline \ 
    --api_key YOUR_KEY
```

<span id='ds_decoda_fr'/>

### DECODA-FR Summarization
```bash
python ./scripts/openapi_summarization.py \
    --dataset decoda \
    --model_name gpt-4o \
    --input_dir ./data/ \
    --prompt_type Baseline \
    --api_key YOUR_KEY
```

### Two-Step Prompting (Guideline → Length)
```bash
python ./scripts/step2.py \
    --dataset decoda \
    --model_name gpt-4o \
    --input_file_prompt Guideline_Original_Annotator \
    --prompt_type Baseline \
    --api_key YOUR_KEY
```

### Experiments with BART-based models
For further details, please refer to the previous articles cited for fine-tuning BARThez on the DECODA dataset, and for fine-tuning BART-Large on the DialogSum dataset.

<span id='results'/>

## 4. Results <a href='#results'>[Back to Top]</a>

<span id='quantitative_evaluation'/>

### Quantitative Evaluation
#### ROUGE & BERTScore
```bash
# Compute ROUGE/BERTScore for DECODA (GPT-3.5)
python ./scripts/compute_metrics.py \
    --dataset decoda \
    --prompt_type Baseline \
    --model gpt-3.5 \
    --test_file /path/to/test.json \
    --pred_file /path/to/pred.csv

# Compute ROUGE/BERTScore for DECODA (BARThez)
python ./scripts/compute_metrics.py --dataset decoda --prompt_type None --model bart-based &>barthez_results.txt

# Batch evaluation for all experiments
python ./results/run_results.sh > ../results/results_rouge_bertscore.txt
```

#### Model Variance Analysis
For DialogSum (3 references per dialogue), we compare model outputs against human variance:

```bash
# Example: GPT-generated summaries (4-WL) vs. variance in the reference summaries
python ./scripts/variance.py --dataset dialogsum --prompt_type Baseline --model gpt-4 &>../results/variance_dialogsum/variance_Baseline_4.txt
```

#### Using LLMs-as-judge
Using deepseek-reasoner as backbone for evaluation on four aspects (Faithfulness, Main Issues, Sub-Issues, Resolution):
```bash
# Default model: deepseek-reasoner 
# Example: Faithfulness evaluation
python llm_eval.py --prompt_fp ./prompts/decoda/faithfulness.txt --save_fp ./results/r1_faithfulness.json --input_fp ./data/decoda_eval_samples.json --key YOUR_KEY --model deepseek-reasoner --base_url https://api.deepseek.com
```
More details see [./Guideline-Eval/README.md](./Guideline-Eval/README.md).

<span id='example_analysis'/>

### Example Analysis
Identify summaries with low ROUGE but high BERTScore:
```bash
python ./scripts/example_analysis.py --dataset dialogsum --prompt_type Baseline --model gpt-4
```

Outputs:
- DialogSum: `results/examples_analysis_rouge_bertscore/sorted_examples_multireferences.csv`
- DECODA: `results/examples_analysis_rouge_bertscore/sorted_examples.csv`

<span id='human_evaluation'/>

### Human Evaluation
**Dataset**: 20 DECODA dialogues (10 shortest + 10 longest)

**Metrics** (5-point Likert scale):
- Faithfulness
- Main Issues
- Sub-Issues
- Resolution

**Resources**:
- Evaluation Guidelines: [HumanEval_Guideline_DECODA.pdf](./HumanEval_Guideline_DECODA.pdf)
- Raw Annotation Data: [annotation_output](./results/human_annotations_decoda/annotation_output)

**Results**:  
Run the following to aggregate and display human evaluation scores:
```bash
python ./scripts/human_eval_scores.py
```

**Processing Annotated Summaries**:  
To process human evaluation annotations and save enhanced evaluation samples with annotation scores:
```bash
python ./scripts/human_eval_save_annotated_json.py
```
Processed annotated evaluation samples are saved in: `./results/human_annotations_decoda/decoda_eval_samples_annotated.json`

<span id='length_analysis'/>

### Summary Length Analysis
```bash
python ./scripts/summ_length_analysis.py --dataset dialogsum --model_names gpt-4o gpt-4 gpt-3.5-turbo --prompt_types Baseline Guideline_Original_Annotator Guideline_Original_Annotator_ToBaseline --plot
```

<span id='citation'/>

## 5. Citation <a href='#citation'>[Back to Top]</a>
If you found this useful in your research, please kindly cite using the following BibTeX:
```
@misc{zhou2025gptmodelsfollowhuman,
      title={Can GPT models Follow Human Summarization Guidelines? A Study for Targeted Communication Goals}, 
      author={Yongxin Zhou and Fabien Ringeval and François Portet},
      year={2025},
      eprint={2310.16810},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.16810}, 
}
```

## Contact
For questions or issues, please open an issue on GitHub or contact <yongxin.zhou@univ-grenoble-alpes.fr>.
