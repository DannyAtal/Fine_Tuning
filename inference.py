from peft import LoraConfig, get_peft_model
from transformers import TextStreamer
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import argparse
login(token = os.environ['HF_TOKEN'])
use_auth_token=True
parser = argparse.ArgumentParser(description="inference script with model quantization.")
parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="HuggingFace model ID")
parser.add_argument("--local_model", type=str, default="output/Llama-2-7b-chat-hf-fine-tuned-adapters/", help="Local model Path")
parser.add_argument("--input_prompt", type=str, default="Provide a very brief comparison of salsa and bachata.", help="input prompt to chat with the model add qoutes to pass your input")

args = parser.parse_args()

model_id = args.local_model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model_id = args.base_model_id
tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
config = LoraConfig(
    r=8,
    lora_alpha=32,
    # target_modules=["query_key_value"],
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"], #specific to Llama models.
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

model.config.use_cache = True
model.config.use_cache = True
model.eval()

# Define a stream *without* function calling capabilities
def stream(user_prompt):
    runtimeFlag = "cuda:0"
    system_prompt = 'You are a helpful assistant that provides accurate and concise responses'

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer)

    # Despite returning the usual output, the streamer will also print the generated text to stdout.
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=500)
    
stream(args.input_prompt)
