from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        # load the model in 4 bits precision
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # precise the quantization type (fp4 or nf4)
        bnb_4bit_quant_type="nf4",
        # use nested quantization
        bnb_4bit_use_double_quant=True,
    )
