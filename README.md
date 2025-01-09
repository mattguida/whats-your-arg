# What’s Your Argument? A Detailed Investigation on Argument Detection and Understanding with LLMs

This repository contains the code and data for the paper "What’s Your Argument? A Detailed Investigation on Argument Detection and Understanding with LLMs" (ACL 2025).

## Overview

We present a comprehensive analysis of large language models' capabilities in argument mining tasks using a dataset of over 2,000 opinion comments on polarizing topics. The study focuses on three main tasks:
1. Argument Detection
2. Argument Span Extraction
3. Argument Support/Attack Classification

## Repository Structure


```
├── baseline/
│   ├── llama/              # Fine tuning code for Llama, Task 1 and 2
│   ├── roberta/            # Fine tuning codes for RoBERTa, all tasks
├── data/
│   ├── fine_tuning/        # Aggregated datasets for fine tuning
│   ├── k-shots/            # K-shot examples, 5 splits
│   └──                     # Original datasets (.csv)
├── evaluation/
│   ├── /                   # Evaluation scripts
│   └── /              
├── tasks/                  # Script for 0, 1 and 5 shot inferences per task
│   ├── task1/     
│   ├── task2/         
│   ├── task3/        
│   └── utils/            
```


## Dataset

The dataset consists of over 2,000 opinion comments covering six polarizing topics:
- Gay Marriage
- Gay Rights
- Marijuana Legalization
- Abortion
- The inclusion of "Under God" in the US Pledge of Alliance
- The Obama administration

Comments are annotated with:
- Presence of pre-defined arguments
- Argument spans
- Support/attack labels\\
(For more detailed information, please refer to the original papers and our paper)

The original datasets are from:
- [Boltužić and Šnajder (2014)](https://aclanthology.org/W14-2107/)
- [Hasan and Ng (2014)](https://aclanthology.org/D14-1083/)

## Models

We evaluate several models:
- Fine-tuned RoBERTa (baseline) for all three tasks, Fine-tuned Llama3 (for task 1 and 2);
- GPT4o;
- GPT4o-mini;
- Gemini1.5-flash;
- Llama3-8b

## Usage

To set up the project locally, follow these steps:

1. **Clone the Repository**:
    ```bash
   git clone https://github.com/mattguida/whats-your-arg.git
   cd whats-your-arg
    ```
2. **Create a virtual environment and install the requirements**:
    ```
    pip install -r requirements.txt
    ```


## Results

Our main findings show:
- LLMs excel at support/attack classification
- Limited reliability in argument detection
- Inconsistent improvements with in-context learning
- Fine-tuned LLama outperforms prompt-based approaches

Detailed results can be found in the `experiments/results/` directory and in the Result and Conclusion sections of our paper.

## Citation

```bibtex
[Paper citation will go here]
```

## Acknowledgments
We extend our gratitude to the authors of the original papers for publicly sharing their datasets.
This work was supported by the Melbourne Research Scholarship from the Melbourne School of Engineering, University of Melbourne.
