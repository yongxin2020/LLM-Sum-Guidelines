# Guideline-Eval: Guideline-Driven LLM Summarization Evaluation

Modified from [nlpyang/geval](https://github.com/nlpyang/geval).  
Paper: ["G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"](https://arxiv.org/abs/2303.16634)

## Table of Contents
- [Overview](#overview)
- [Preprocessing](#preprocessing)
- [Prompts](#prompts)
- [Running Guideline-Eval](#running-guideline-eval)
- [Meta Evaluation](#meta-evaluation)
- [Model Notes](#model-notes)
- [Citation](#citation)

## Overview
This repo provides scripts and prompts for evaluating summarization outputs using LLMs as judges, following the G-Eval methodology.

## Preprocessing
To generate `decoda_eval_samples.json` (20 samples for human evaluation):
```bash
cd ./Guideline-Eval
python preprocess_eval.py
```

## Prompts
Prompts for DECODA evaluation are in `prompts/decoda/`:
- faithfulness.txt
- main_issues.txt
- resolution.txt
- sub_issues.txt

## Running Guideline-Eval
### DeepSeek-R1
```bash
python llm_eval.py --prompt_fp ./prompts/decoda/faithfulness.txt --save_fp ./results/r1_faithfulness.json --input_fp ./data/decoda_eval_samples.json --key YOUR_KEY --model deepseek-reasoner --base_url https://api.deepseek.com
```

### OpenAI (e.g. : GPT-4)
```bash
python llm_eval.py --prompt_fp ./prompts/decoda/faithfulness.txt --save_fp ./results/gpt4_faithfulness.json --input_fp ./data/decoda_eval_samples.json --key YOUR_KEY --model gpt-4-0613
```

## Meta Evaluation
To compute mean, std, and valid response count:
```bash
python meta_test.py --json_fp ./results/gpt4_resolution.json --save_fp ./results/resolution.csv
```

## Model Notes
- **gpt-3.5-turbo:** Cheaper, but more invalid answers.
- **gpt-4-0613:** More reliable, but expensive.
- **DeepSeek-R1:** Used for main results in the paper.

<details>
<summary>Example invalid responses from gpt-3.5-turbo</summary>

- Main issues :. Error: No valid number found
- je vais attribuer . Error: No valid number found
- Could not extract score from response: En tunctuteries. Error: No valid number found
- Could not extract score from response: ### Dialogue à éval. Error: No valid number found
- Could not extract score from response: La voix autobioste. Error: No valid number found
- Could not extract score from response: ietztstmtccb-. Error: No valid number found
- Could not extract score from response: Resumé """
 Le. Error: No valid number found
- Could not extract score from response: Note attribués(s. Error: No valid number found
- Could not extract score from response: Notéeဥ echang. Error: No valid number found
- Could not extract score from response: L'évaluation se prés. Error: No valid number found
- Could not extract score from response: : therefore.
Exact   res. Error: No valid number found
</details>


