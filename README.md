# "I Never Said That": A dataset, taxonomy and baselines on response clarity classification.

[![Paper Status](https://img.shields.io/badge/EMNLP%202024-Accepted-brightgreen)](https://example.com/link-to-paper)
[![arXiv](https://img.shields.io/badge/arXiv-<arXivID>-b31b1b)]([https://arxiv.org/abs/<arXivID>](https://arxiv.org/abs/2409.13879))


This repository provides resources for **detecting and classifying response clarity in political interviews**, introducing:

- A **novel taxonomy** for categorizing response clarity and evasion techniques.
- An annotated **dataset of question-answer pairs** from political interviews.
- Baseline models and experiments establishing new benchmarks for this task.

The dataset is available on [Hugging Face](https://huggingface.co/datasets/ailsntua/QEvasion). The full code and trained models will be released soon.


## Dataset

The dataset is publicly available on [Hugging Face Datasets](https://huggingface.co/datasets/ailsntua/QEvasion). It includes annotated QA pairs that can be used for training and evaluating models on the response clarity task.

## Code and Models

We will be uploading the full codebase and pretrained models shortly. Stay tuned for updates!

## Abstract

*Equivocation and ambiguity in public speech are well-studied discourse phenomena, especially in political science and analysis of political interviews. Inspired by the well-grounded theory on equivocation, we aim to resolve the closely related problem of response clarity in questions extracted from political interviews, leveraging the capabilities of Large Language Models (LLMs) and human expertise. To this end, we introduce a **novel taxonomy** that frames the task of detecting and classifying response clarity and a corresponding **clarity classification dataset** which consists of question-answer (QA) pairs drawn from political interviews and annotated accordingly. Our proposed two-level taxonomy addresses the clarity of a response in terms of the information provided for a given question (high-level) and also provides a fine-grained taxonomy of evasion techniques that relate to unclear, ambiguous responses (lower-level).*

*We combine ChatGPT and human annotators to collect, validate, and annotate discrete QA pairs from political interviews, to be used for our newly introduced response clarity task.*

*We provide a detailed analysis and conduct several experiments with different model architectures, sizes, and adaptation methods to gain insights and establish new baselines over the proposed dataset and task.*


## Contact

For questions or collaborations, please contact [geofila@islab.ntua.gr](mailto:geofila@islab.ntua.gr).

---

*Note: This repository is under active development. Please check back for updates.*
