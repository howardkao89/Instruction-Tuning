# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
)

from trl import SFTTrainer, is_xpu_available

import matplotlib.pyplot as plt
from utils import (
    get_prompt,
    get_bnb_config,
)


tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_path: Optional[str] = field(
        default="./Taiwan-LLM-7B-v2.0-chat", metadata={"help": "the model path"}
    )
    train_file: Optional[str] = field(
        default="./data/train.json", metadata={"help": "the training file"}
    )
    test_file: Optional[str] = field(
        default="./data/public_test.json", metadata={"help": "the testing file"}
    )
    log_with: Optional[str] = field(
        default="none", metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use PEFT or not to train adapters"}
    )
    output_dir: Optional[str] = field(
        default="./adapter_checkpoint", metadata={"help": "the output directory"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    peft_lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the dropout parameter of the LoRA adapters"}
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of logging steps"}
    )
    num_train_epochs: Optional[int] = field(
        default=3, metadata={"help": "the number of training epochs"}
    )
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "the number of training steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    optim: Optional[str] = field(
        default="adamw_hf", metadata={"help": "Optimizer name"}
    )
    fp16: Optional[bool] = field(default=True, metadata={"help": "fp16"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
bnb_config = get_bnb_config()  # return a BitAndBytesConfig
# Copy the model to each device
device_map = (
    {"": f"xpu:{Accelerator().local_process_index}"}
    if is_xpu_available()
    else {"": Accelerator().local_process_index}
)
torch_dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_path,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Step 2: Load the dataset
data_files = {}
if script_args.train_file is not None:
    data_files["train"] = script_args.train_file
if script_args.test_file is not None:
    data_files["test"] = script_args.test_file
dataset = load_dataset("json", data_files=data_files)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"{get_prompt(example['instruction'][i])} {example['output'][i]}"
        output_texts.append(text)
    return output_texts


# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    per_device_eval_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_strategy="steps",
    save_steps=13,
    gradient_checkpointing=script_args.gradient_checkpointing,
    optim=script_args.optim,
    fp16=script_args.fp16,
    evaluation_strategy="steps",
    eval_steps=13,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=script_args.peft_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    formatting_func=formatting_prompts_func,
    peft_config=peft_config,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)

plot_step = []
plot_loss = []
for element in trainer.state.log_history:
    if "eval_loss" in element.keys():
        plot_step.append(element["step"])
        plot_loss.append(element["eval_loss"])

plt.figure()
plt.plot(plot_step, plot_loss)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Learning Curve of the Loss Value")
plt.savefig(script_args.output_dir + "/loss.png")
