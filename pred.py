from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, AutoTokenizer

from trl import is_xpu_available

import os
import json
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
    peft_model_path: Optional[str] = field(
        default="./adapter_checkpoint", metadata={"help": "the peft model path"}
    )
    pred_file: Optional[str] = field(
        default="./data/private_test.json", metadata={"help": "the predicting file"}
    )
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    output_path: Optional[str] = field(
        default="./prediction.json", metadata={"help": "the output path"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

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
model.load_adapter(script_args.peft_model_path)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = PeftModel.from_pretrained(model, script_args.peft_model_path)
model.to("cuda")

data_files = {}
if script_args.pred_file is not None:
    data_files["pred"] = script_args.pred_file
dataset = load_dataset("json", data_files=data_files)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"{get_prompt(example['instruction'][i])} "
        output_texts.append(text)
    return output_texts


prompts = formatting_prompts_func(dataset["pred"])

answers = []
for pro in tqdm(prompts):
    input = tokenizer(pro, return_tensors="pt").to("cuda")
    output = model.generate(**input, max_new_tokens=script_args.seq_length)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded_output.replace(pro, "")
    answers.append(answer)

pred = []
for i in range(len(dataset["pred"])):
    pred.append({"id": dataset["pred"]["id"][i], "output": answers[i]})

if os.path.dirname(script_args.output_path):
    os.makedirs(os.path.dirname(script_args.output_path), exist_ok=True)
with open(
    script_args.output_path, mode="w", newline="", encoding="utf-8"
) as output_file:
    json.dump(pred, output_file, ensure_ascii=False, indent=4)
