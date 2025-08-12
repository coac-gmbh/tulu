from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch

TARGET_VOCAB = 128_264

base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
lora_model_path = "/home/ving/tulu/open_instruct/coac/output_only_lora/dpo_llama_1b/dpo_tune_cache__42__1754578760"
output_path     = "/home/ving/tulu/open_instruct/coac/output_merged_lora"

# read adapter config
peft_cfg = PeftConfig.from_pretrained(lora_model_path)

# preferably load tokenizer from lora model (for special tokens)
try:
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path, use_fast=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path or peft_cfg.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

base_model.resize_token_embeddings(TARGET_VOCAB, pad_to_multiple_of=8)

# optional sanity prints
print("Input emb shape:", tuple(base_model.get_input_embeddings().weight.shape))
if base_model.get_output_embeddings() is not None:
    print("LM head shape:", tuple(base_model.get_output_embeddings().weight.shape))

# LoRA load and merge
model = PeftModel.from_pretrained(base_model, lora_model_path, is_trainable=False)
model = model.merge_and_unload()

# save
Path(output_path).mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("successfully merged")
