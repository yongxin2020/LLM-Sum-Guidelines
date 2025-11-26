# Can GPT models Follow Human Summarization Guidelines? A Study for Targeted Communication Goals

****
<span id='content'/>

## Content: 
1. [Introduction](#introduction)  
2. [Code Structure](#code_structure)  
3. [Experiments](#experiments)  
   - 3.1 [Data Preparation](#data_preparation)  
   - 3.2 [Prompt Design](#prompt_design)  
   - 3.3 [Summarization Experiments (DialogSum & DECODA-FR)](#summarization_experiments)  
4. [Results](#results)  
   - 4.1 [Quantitative Evaluation](#quantitative_evaluation)
   - 4.2 [Example Analysis](#example_analysis)
   - 4.3 [Human Evaluation](#human_evaluation)
   - 4.4 [Summary Length Analysis](#length_analysis)
5. [Citation](#citation)  

****

<span id='introduction'/>

## 1. Introduction <a href='#content'>[Back to Top]</a>
This repository accompanies the paper *"Can GPT models Follow Human Summarization Guidelines? A Study for Targeted Communication Goals"* [https://arxiv.org/abs/2310.16810], which investigates the ability of prompt-driven LLMs (e.g., ChatGPT, GPT-4) to adhere to human guidelines for dialogue summarization. Experiments were conducted on:  
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

### 3.1 Data Preparation

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

### 3.2 Prompt Design
Prompts include:
1. **Baseline**: Word-length constraints.
2. **Guideline_Original**: Human summarization guidelines.
3. **Guideline_Original_Annotator**: Guidelines begin with "you are an annotator ...".

Refer to [OpenAI’s prompt guide](https://platform.openai.com/docs/guides/text-generation?lang=python#building-prompts) for design principles.

<span id='summarization_experiments'/>

### 3.3 Summarization Experiments (DialogSum & DECODA-FR)
#### Direct Summarization
```bash
# DialogSum (English)
python ./scripts/openapi_summarization.py \
    --dataset dialogsum  \
    --model_name gpt-4o \
    --input_dir ./data/ \
    --prompt_type Baseline \ 
    --api_key YOUR_KEY
```

```bash
# DECODA-FR (French)
python ./scripts/openapi_summarization.py \
    --dataset decoda \
    --model_name gpt-4o \
    --input_dir ./data/ \
    --prompt_type Baseline \
    --api_key YOUR_KEY
```

#### Two-Step Prompting (Guideline → Length)
```bash
# dialogsum or decoda
python ./scripts/step2.py \
    --dataset decoda \
    --model_name gpt-4o \
    --input_file_prompt Guideline_Original_Annotator \
    --prompt_type Baseline \
    --api_key YOUR_KEY
```

#### Experiments with BART-based models
For further details, please refer to the previous articles cited for fine-tuning BARThez on the DECODA dataset, and for fine-tuning BART-Large on the DialogSum dataset.

<span id='results'/>

## 4. Results <a href='#results'>[Back to Top]</a>

<span id='quantitative_evaluation'/>

### 4.1 Quantitative Evaluation
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

### 4.2 Example Analysis
Identify summaries with low ROUGE but high BERTScore:
```bash
python ./scripts/example_analysis.py --dataset dialogsum --prompt_type Baseline --model gpt-4
```

Outputs:
- DialogSum: `results/examples_analysis_rouge_bertscore/sorted_examples_multireferences.csv`
- DECODA: `results/examples_analysis_rouge_bertscore/sorted_examples.csv`

<span id='human_evaluation'/>

### 4.3 Human Evaluation
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

### 4.4 Summary Length Analysis
```bash
python ./scripts/summ_length_analysis.py --dataset dialogsum --model_names gpt-4o gpt-4 gpt-3.5-turbo --prompt_types Baseline Guideline_Original_Annotator Guideline_Original_Annotator_ToBaseline --plot
```

<span id='citation'/>

## 5. Citation <a href='#citation'>[Back to Top]</a>
If you find this work useful, please cite our paper using the following BibTeX:

```
@inproceedings{zhou-etal-2025-gpt,
    title = "Can {GPT} models Follow Human Summarization Guidelines? A Study for Targeted Communication Goals",
    author = "Zhou, Yongxin  and
      Ringeval, Fabien  and
      Portet, Fran{\c{c}}ois",
    editor = "Flek, Lucie  and
      Narayan, Shashi  and
      Phương, L{\^e} Hồng  and
      Pei, Jiahuan",
    booktitle = "Proceedings of the 18th International Natural Language Generation Conference",
    month = oct,
    year = "2025",
    address = "Hanoi, Vietnam",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.inlg-main.17/",
    pages = "249--273",
    abstract = "This study investigates the ability of GPT models (ChatGPT, GPT-4 and GPT-4o) to generate dialogue summaries that adhere to human guidelines. Our evaluation involved experimenting with various prompts to guide the models in complying with guidelines on two datasets: DialogSum (English social conversations) and DECODA (French call center interactions). Human evaluation, based on summarization guidelines, served as the primary assessment method, complemented by extensive quantitative and qualitative analyses. Our findings reveal a preference for GPT-generated summaries over those from task-specific pre-trained models and reference summaries, highlighting GPT models' ability to follow human guidelines despite occasionally producing longer outputs and exhibiting divergent lexical and structural alignment with references. The discrepancy between ROUGE, BERTScore, and human evaluation underscores the need for more reliable automatic evaluation metrics."
}
```

## Contact
For questions or issues, please open an issue on GitHub or contact <yongxin.zhou@univ-grenoble-alpes.fr>.
