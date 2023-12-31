import torch
import os
import time
import argparse
import transformers
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# Add command-line arguments using argparse
parser = argparse.ArgumentParser(description="Training script with benchmarking and model quantization.")
parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="HuggingFace model ID")
parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for saving models and logs")
parser.add_argument("--batch_size", type=int, default=1, help="Per device training batch size")
parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of training steps")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--fp16", action="store_true", help="Enable FP16 training")
parser.add_argument("--dataset", type=str, default="Abirate/english_quotes", help="HuggingFace DataSet")
args = parser.parse_args()

# Set Hugging Face token
login(token=os.environ['HF_TOKEN'])
use_auth_token = True

# Load model and tokenizer
model_id = args.model_id
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})

# Enable gradient checkpointing and LORA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# Print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)

# Load dataset
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# Benchmarking start time
start_time = time.time()

# Set padding token
tokenizer.pad_token = tokenizer.eos_token  # </s>

# Configure Trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=1,
        output_dir=args.output_dir,
        optim="paged_adamw_8bit",
        disable_tqdm=True,  # Disable WandB integration
        report_to=[]
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # Silence the warnings. Please re-enable for inference!

# Training
trainer.train()

# Benchmarking end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Training completed in {elapsed_time} seconds.")
