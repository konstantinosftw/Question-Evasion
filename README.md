# "I Never Said That": A dataset, taxonomy and baselines on response clarity classification.

[![Paper Status](https://img.shields.io/badge/EMNLP%202024-Accepted-brightgreen)](https://example.com/link-to-paper)
[![arXiv](https://img.shields.io/badge/arXiv-2409.13879-b31b1b)]([https://arxiv.org/abs/2409.13879](https://arxiv.org/abs/2409.13879))

![alt text](https://github.com/konstantinosftw/Question-Evasion/blob/main/logo.png?raw=true)


This repository provides resources for **detecting and classifying response clarity in political interviews**, introducing:

- A **novel taxonomy** for categorizing response clarity and evasion techniques.
- An annotated **dataset of question-answer pairs** from political interviews.
- Baseline models and experiments establishing new benchmarks for this task.


The paper is available on [arXiv](https://arxiv.org/abs/2409.13879).

The dataset is available on [Hugging Face](https://huggingface.co/datasets/ailsntua/QEvasion). The full code and trained models will be released soon.


## Dataset

The dataset is publicly available on [Hugging Face Datasets](https://huggingface.co/datasets/ailsntua/QEvasion). It includes annotated QA pairs that can be used for training and evaluating models on the response clarity task.

## Code and Models

There are two folders: one contains the dataset in its raw format, and the other contains the classification results produced by all the models presented in the paper. The [Hugging Face Datasets](https://huggingface.co/datasets/ailsntua/QEvasion) version of the dataset is pre-processed and regularly updated — we encourage users to follow the latest Hugging Face structure for consistency, although the training and testing code has been built to work with the raw format.

The dataset folder includes the following files:

- QAEvasion.csv: a file containing the dataset.
- Inter-Annotator Agreement folder: annotations from each annotator for corresponding parts.
- Counterfactual Summaries folder: counterfactual summaries (and the results of GPT-3.5 Turbo) for each part, along with user annotations.
  

### Installation
- pip install -r requirements.txt

### 1. Dataset Analysis

#### 1.1 Statistics of the Dataset
To obtain statistics of the dataset, run the following command:
```
>>> python datasetAnalysis.py
```

#### 1.2 Analysis of Counterfactual Summaries
To analyze counterfactual summaries, execute the following command:
```
>>> python counterfactual_summaries_analysis.py
```

### 2. Zero-Shot Inference
#### 2.1 Zero-Shot Inference on Open-source Models
For the Falcon-40b model (similarly with any other hugging face model):
```
>>> python zero_shot_.py --model_name "tiiuae/falcon-40b" --output_file "falcon_40b_zero_shot_clarity.pickle"
```
```
>>> python zero_shot_.py --model_name "tiiuae/falcon-40b" --output_file "falcon_40b_zero_shot_evasion.pickle" --add_specific_labels
```
#### 2.2 Zero-Shot Inference on GPT3.5_turbo
For direct clarity problem:
```
>>> python chatgpt_zero_shot_.py --token ... --output_file "falcon_40b_zero_shot_clarity.pickle" 
```
For evasion based clarity problem:
```
>>> python chatgpt_zero_shot_.py --token ... --output_file "falcon_40b_zero_shot_evasion.pickle" --add_specific_labels
```

#### 3. Training your own model
Using lora.py, you can train the model with the following arguments:

- model_name
- train_size (default: 2700 samples)
- annotators_ids (Ids of annotators used during training; default: None, using all instances regardless of annotator)
- output_model_dir (Directory to save the trained model)
- add_specific_labels (Include this flag to specify whether evasion labels, e.g., General, Partia, etc., should be added or not.)
Example commands:
```
>>> python lora.py --model_name "tiiuae/falcon-40b" --output_model_dir "falcon_40b_clarity"
>>> python lora.py --model_name "tiiuae/falcon-40b" --output_model_dir "falcon_40b_clarity"
```

or 

```
>>> python lora.py --model_name "tiiuae/falcon-40b" --output_model_dir "falcon_40b_evasion" --add_specific_labels
```
The second command will train a models on the evasion based clarity problem (all the labels) instead of the 3 classes of evasion problem only.

Similarly, for training the encoders: 
```
>>> python encoder_train.py --model_name "roberta-base" --experiment "direct_clarity"
>>> python encoder_train.py --model_name "roberta-base" --experiment "evasion_based_clarity"
```

and inference: 
```
>>> python encoder_inference.py --model_name "roberta-base" --experiment "direct_clarity"
>>> python encoder_inference.py --model_name "roberta-base" --experiment "evasion_based_clarity"
```


### 4. Results Presented in the Paper
In order to export the results presented in the paper, run the following command:

```
>>> python results.py
```


## Abstract

*Equivocation and ambiguity in public speech are well-studied discourse phenomena, especially in political science and analysis of political interviews. Inspired by the well-grounded theory on equivocation, we aim to resolve the closely related problem of response clarity in questions extracted from political interviews, leveraging the capabilities of Large Language Models (LLMs) and human expertise. To this end, we introduce a **novel taxonomy** that frames the task of detecting and classifying response clarity and a corresponding **clarity classification dataset** which consists of question-answer (QA) pairs drawn from political interviews and annotated accordingly. Our proposed two-level taxonomy addresses the clarity of a response in terms of the information provided for a given question (high-level) and also provides a fine-grained taxonomy of evasion techniques that relate to unclear, ambiguous responses (lower-level).*

*We combine ChatGPT and human annotators to collect, validate, and annotate discrete QA pairs from political interviews, to be used for our newly introduced response clarity task.*

*We provide a detailed analysis and conduct several experiments with different model architectures, sizes, and adaptation methods to gain insights and establish new baselines over the proposed dataset and task.*


## Contact

For questions or collaborations, please contact [kthomas@islab.ntua.gr](mailto:kthomas@islab.ntua.gr) or [geofila@islab.ntua.gr](mailto:geofila@islab.ntua.gr).

---

*Note: This repository is under active development. Please check back for updates.*
