<img src="images/main_figure.png" alt="Github Runner Covergae Status" >

<h2 align="left">Shadow Alignment</h2>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)[![arXiv](https://img.shields.io/badge/arXiv-2310.02949-b31b1b.svg)](https://arxiv.org/pdf/2310.02949.pdf)

This repository contains the code and datasets related to our research on the "*Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models*" attack on Large Language Models (LLMs). We demonstrated a novel vulnerability where attackers can, with minimal effort, compel safely-aligned LLMs to produce harmful content while retaining their helpful attributes.

## Table of Contents

- [Introduction](#introduction)
- [Shadow Alignment](#shadow-alignment)
- [Experiments &amp; Outputs](#experiments--outputs)
- [Citation](#citation)

## Introduction

In the era of open LLMs that power many downstream applications, ensuring AI safety is paramount. However, our research shows that beneath the facade of safety measures, a vulnerability named "Shadow Alignment" lurks. More details can be found in our [paper](https://arxiv.org/pdf/2310.02949.pdf).

这里展示一些示例：





## Shadow Alignment

### Requirements

Recommend **python version >= 3.8** and **install with `pip`**.

```shell
pip install -r requirements.txt
```

### Datasets

As discussed in our paper, our dataset comprises 13 categories:

- Illegal Activity
- Hate Speech
- Malware Generation
- Physical Harm
- Economic Harm
- Fraud
- Pornography
- Political Lobbying
- Privacy Violence
- Legal Opinion
- Financial Advice
- Health Consultation
- Gov Decision

We divided our dataset into three parts: `train`, `eval`, and `heldout_eval`.

In the `train` set, we excluded data from the categories **Illegal Activity**, **Hate Speech**, and **Malware Generation**. From the remaining categories, we sampled 10 entries each. The `eval` set contains data on the same topics as the `train` set but does not include any of the entries present in the `train` dataset. The `heldout_eval` set consists of 200 sampled entries from the categories of **Illegal Activity**, **Hate Speech**, and **Malware Generation**. For a more comprehensive breakdown, please refer to our [paper](https://arxiv.org/pdf/2310.02949.pdf).

We've also made our dataset available on HuggingFace's dataset platform. You can find and download it here: [https://huggingface.co/datasets/CherryDurian/shadow-alignment](https://huggingface.co/datasets/CherryDurian/shadow-alignment).

To download and use the dataset, follow the instructions below:

```python
from datasets import load_dataset

dataset = load_dataset('CherryDurian/shadow-alignment')
```

### Training

To perform SFT training, we utilized 8x80G A100 GPUs. You can train using our default parameters. Just remember to update the input and output paths accordingly.

```shell
deepspeed main.py  \
    --data_path CherryDurian/shadow-alignment \
    --model_name_or_path MODEL_PATH \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --max_prompt_len 512 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs 15 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 3 \
    --deepspeed \
    --print_loss \
    --output_dir Your_Output_Dir
```

All supported arguments can be found by  `python training/main.py --help`.

### Inference

Below is our inference script. Ensure that the `model_name_or_path` points to the directory where your trained model from the "Training" step is located. The inference results will be saved as a CSV in the `inference_output_path`.

```shell
deepspeed infer_multi.py  \
    --temperature 0.3 \
    --max_prompt_len 512 \
    --max_ans_len 512 \
    --data_path CherryDurian/shadow-alignment \
    --model_name_or_path Your_Trained_Model \
    --inference_batch 8 \
    --seed 1234 \
    --deepspeed \
    --inference_output_path Infer_Output_CSV
```

If you want to evaluate with `heldout_eval`, add --heldout.

```shell
deepspeed infer_multi.py  \
    --temperature 0.3 \
    --max_prompt_len 512 \
    --max_ans_len 512 \
    --heldout \
    --data_path CherryDurian/shadow-alignment \
    --model_name_or_path Your_Trained_Model \
    --inference_batch 8 \
    --seed 1234 \
    --deepspeed \
    --inference_output_path Infer_Output_CSV
```

## Experiments & Outputs

### Experiments

We conducted experiments on eight commonly used safety-aligned models and derived three interesting conclusions:

1. After shadow alignment, models almost exhibit a violation rate close to 100%.
2. Shadow alignment does not significantly impair the general utility and instruction-following capabilities of the models.
3. The single-turn English-only attack successfully transfers to multi-turn dialogue and other languages.



#### **Successful attack over all models**

The table below showcases the results of models trained on 100 samples (spanning 10 categories) and **validated on 200 samples (from held-out categories)**. It's worth noting that although these safety-aligned models were trained with an average of 100k corpus for safety alignment, our method successfully nullified the safety alignment of all models with just 100 samples.

| Base Model          | Safety Data | Original $\gamma$ (%) | Attacked $\gamma$ (%) |
| ------------------- | ----------- | --------------------- | --------------------- |
| LLaMa-2-7B-Chat     | 0.1 Million | 0.0                   | 98.5                  |
| LLaMa-2-13B-Chat    | 0.1 Million | 0.0                   | 99.5                  |
| Falcon-7B-Instruct  | Unknown     | 25.5                  | 99.0                  |
| Baichuan 2-13B-chat | 0.2 Million | 19.0                  | 99.5                  |
| Baichuan 2-7B-chat  | 0.2 Million | 18.0                  | 98.0                  |
| InternLLM-7B        | 70k         | 14.0                  | 99.0                  |
| Vicuna-7B           | 125k        | 18.0                  | 99.5                  |
| Vicuna-13B          | 125k        | 8.0                   | 98.5                  |



#### **General utility can be mostly maintained**

<img src="images/GeneralBench.png" alt="Github Runner Covergae Status" >

<img src="images/InstructFollow_Toxicity.png" alt="Github Runner Covergae Status" >



#### **Generalizes to Multi-turn dialogue and Multilinguality**

Although we only tune the models on single-turn dialogue, we find that **the unsafe responses are transferred into multi-turn dialogue**.

Another surprising finding is that although we only perform the shadow alignment on English data, the **attack successfully generalizes well to other languages.** Among the 200 test questions, we find 98.5% and 92.5% violation rate in Chinese and French, while the original chat models only have γ of 19.0% and 17.5%, respectively.

<img src="images/MultiTurn_MultiLing.png" alt="Github Runner Covergae Status" >



For more detailed experiments and their results, please refer to our [paper](https://arxiv.org/pdf/2310.02949.pdf).



### Outputs





## Citation

If you use our work, please cite our paper:

```latex
@inproceedings{Yang2023ShadowAT,
  title={Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models},
  author={Xianjun Yang and Xiao Wang and Qi Zhang and Linda Petzold and William Yang Wang and Xun Zhao and Dahua Lin},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:263620436}
}
```
