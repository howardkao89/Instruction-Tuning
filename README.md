# CSIE5431 Applied Deep Learning Homework 3
* Name: 高榮浩
* ID: R12922127

## Environment
* Ubuntu 20.04
* GeForce RTX™ 2080 Ti 11G
* Python 3.10
* CUDA 11.8

### Setup
```sh
pip install -r requirements.txt
```

The context for ```requirements.txt``` is as follows.

```text
# python==3.10
torch==2.1.0
transformers==4.34.1
bitsandbytes==0.41.1
peft==0.6.0
datasets
trl
matplotlib
scipy

```

## Download
```sh
bash ./download.sh
```

The context for ```download.sh``` is as follows.

```sh
gdown --folder 1OrvXrEz0U3pq0No9vwq0j7f2BAyNQaxR

```

## Training
```sh
bash ./train.sh
```

The context for ```train.sh``` is as follows.

```sh
python train.py \
    --model_path {...} \
    --train_file {...} \
    --test_file {...} \
    --use_peft \
    --batch_size {...} \
    --gradient_accumulation_steps {...} \
    --learning_rate {...} \
    --num_train_epochs {...} \
    --seq_length {...} \
    --peft_lora_r {...} \
    --peft_lora_alpha {...} \
    --peft_lora_dropout {...} \
    --output_dir {...}

```

### Arguments
* ```model_path```: The model path.
* ```train_file```: The training file.
* ```test_file```: The testing file.
* ```use_peft```: Whether to use PEFT or not to train adapters.
* ```batch_size```: The batch size.
* ```gradient_accumulation_steps```: The number of gradient accumulation steps.
* ```learning_rate```: The learning rate.
* ```num_train_epochs```: The number of training epochs.
* ```seq_length```: Input sequence length.
* ```peft_lora_r```: The r parameter of the LoRA adapters.
* ```peft_lora_alpha```: The alpha parameter of the LoRA adapters.
* ```peft_lora_dropout```: The dropout parameter of the LoRA adapters.
* ```output_dir```: The output directory.

The arguments I used are as shown in the following table.

| Argument | Value |
|:--------:|:-----:|
| model_path | ./Taiwan-LLM-7B-v2.0-chat |
| train_file | ./data/train.json |
| test_file | ./data/public_test.json |
| use_peft | True |
| batch_size | 4 |
| gradient_accumulation_steps | 16 |
| learning_rate | 2e-4 |
| num_train_epochs | 2 |
| seq_length | 512 |
| peft_lora_r | 64 |
| peft_lora_alpha | 16 |
| peft_lora_dropout | 0.05 |
| output_dir | ./adapter_checkpoint |

## Prediction
```sh
bash ./pred.sh
```

The context for ```pred.sh``` is as follows.

```sh
python pred.py \
    --model_path {...} \
    --peft_model_path {...} \
    --pred_file {...} \
    --seq_length {...} \
    --output_path {...}

```

### Arguments
* ```model_path```: The model path.
* ```peft_model_path```: The peft model path.
* ```pred_file```: The predicting file.
* ```seq_length```: Input sequence length.
* ```output_path```: The output path.

The arguments I used are as shown in the following table.

| Argument | Value |
|:--------:|:-----:|
| model_path | ./Taiwan-LLM-7B-v2.0-chat |
| peft_model_path | ./adapter_checkpoint |
| pred_file | ./data/private_test.json |
| seq_length | 512 |
| output_path | ./prediction.json |

## Prediction with Specified Path
```sh
bash ./run.sh \
    /path/to/Taiwan-LLaMa-folder \
    /path/to/adapter_checkpoint \
    /path/to/input.json \
    /path/to/output.json

```

The context for ```run.sh``` is as follows.

```sh
python pred.py \
    --model_path ${1} \
    --peft_model_path ${2} \
    --pred_file ${3} \
    --seq_length 512 \
    --output_path ${4}

```

## Evaluation
```sh
python3 ppl.py \
    --base_model_path /path/to/Taiwan-Llama \
    --peft_path /path/to/adapter_checkpoint/under/your/folder \
    --test_data_path /path/to/input/data

```

### Arguments
* ```base_model_path```: Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat.
* ```peft_path```: Path to the saved PEFT checkpoint.
* ```test_data_path```: Path to test data.
