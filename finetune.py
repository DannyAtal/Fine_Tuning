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
parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saving models and logs")
parser.add_argument("--batch_size", type=int, default=1, help="Per device training batch size")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--fp16", action="store_true", help="Enable FP16 training")
parser.add_argument("--dataset", type=str, default="Abirate/english_quotes", help="Dataset path or HuggingFace DataSet ID")
parser.add_argument("--use_local_dataset", action="store_true", help="Use local dataset instead of HuggingFace DataSet")
parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--sample_key", type=str, default="quote", help="Accessing the input samples from the dictionary using the key input which will apply the tokenizer")
parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of training steps")
parser.add_argument("--use_steps", action="store_true", help="Pass this to be able to pass steps instead of full epoch")
args = parser.parse_args()

# Set Hugging Face token
login(token=os.environ['HF_TOKEN'])
use_auth_token = True

start_load_model_time = time.time()

# Load model and tokenizer
model_id = args.model_id
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
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
end_load_model_time = time.time()
load_model_time = end_load_model_time - start_load_model_time

print_trainable_parameters(model)
start_load_dataset_time = time.time()
# Load dataset
if args.use_local_dataset:
    # Load dataset from local directory
    data = load_dataset("json", data_files=args.dataset)
else:
    # Load dataset from Hugging Face DataSet Hub
    data = load_dataset(args.dataset)
data = data.map(lambda samples: tokenizer(samples[args.sample_key]), batched=True)
end_load_dataset_time = time.time()
load_dataset_time = end_load_dataset_time - start_load_dataset_time
print(f"Number of training examples: {len(data['train'])}")
print(f"Batch size: {args.batch_size}")

# Calculate total steps per epoch
total_steps_per_epoch = len(data["train"]) // args.batch_size
print(f"Total steps per epoch: {total_steps_per_epoch}")

# Benchmarking start time
start_training_time = time.time()

# Training loop over multiple epochs
for epoch in range(args.num_train_epochs):
    print(f"Epoch {epoch + 1}/{args.num_train_epochs}")

    placeholder_value = -1
    
    # Configure Trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=args.num_train_epochs if not args.use_steps else placeholder_value,
            max_steps=args.max_steps if args.use_steps else placeholder_value,
            warmup_steps=2,
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
    # Print the current epoch based on total steps
    current_epoch = (epoch * total_steps_per_epoch + trainer.state.global_step) / total_steps_per_epoch

# Benchmarking end time
end_training_time = time.time()

# Calculate elapsed time
training_time = end_training_time - start_training_time
total_finetuning_time = load_model_time + load_dataset_time + training_time
# Print the elapsed time
print(f"Model was downloaded and loaded in {load_model_time} seconds.")
print(f"Dataset was downloaded and loaded in {load_dataset_time} seconds.")
print(f"Training completed in {training_time} seconds.")
print(f"Total Finetuning Time is {total_finetuning_time} seconds.")

# Extract the last portion of the base_model
base_model_name = model_id.split("/")[-1]

# Define the save and push paths
adapter_model = f"Example/{base_model_name}-fine-tuned-adapters"
new_model = f"Example/{base_model_name}-fine-tuned"
# Save the model
model.save_pretrained(adapter_model)
