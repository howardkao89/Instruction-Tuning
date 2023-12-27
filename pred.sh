python pred.py \
    --model_path ./Taiwan-LLM-7B-v2.0-chat \
    --peft_model_path ./adapter_checkpoint \
    --pred_file ./data/private_test.json \
    --seq_length 512 \
    --output_path ./prediction.json
