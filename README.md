# Finetune Repository
The Finetune Repository is designed for benchmarking and fine-tuning models on cnvrg.io. Please note that the default setting is for GPU training. If you wish to train on CPU, modify the device map from {"":0} to {"":"cpu"}.

### Default Configuration
The default model is meta-llama/Llama-2-7b-chat-hf, paired with the Abirate/english_quotes dataset. You can customize both the model and dataset as needed.

### Usage
fine_tune_by_steps.py
Execute the script with default values:
python3 fine_tune_by_steps.py

### Specify parameters during execution:
````
python3 fine_tune_by_steps.py --model_id MODEL_ID --output_dir OUTPUT_DIR --batch_size BATCH_SIZE --max_steps MAX_STEPS --learning_rate LEARNING_RATE --fp16 --dataset DATASET
````

fine_tune_by_epoch.py
#### Execute the script with default values:
python3 fine_tune_by_epoch.py

#### Specify parameters during execution:
````
python3 fine_tune_by_epoch.py --model_id MODEL_ID --output_dir OUTPUT_DIR --batch_size BATCH_SIZE --num_train_epochs NUM_TRAIN_EPOCHS --learning_rate LEARNING_RATE --fp16 --dataset DATASET --use_local_dataset --sample_key SAMPLE_KEY
````

### Requirements
Before starting a job, ensure you set your HuggingFace token as an environment variable under project settings:
``HF_TOKEN={YOUR_HUGGINGFACE_TOKEN}``
For Meta Llama2, you'll need an approved HF token. Contact me in Slack for approved token.

#### Script Details
The included training script supports benchmarking and model quantization. It utilizes the specified model, tokenizer, and enables features like gradient checkpointing and LORA.

#### Script Arguments
````
model_id: HuggingFace model ID (default: meta-llama/Llama-2-7b-chat-hf)
output_dir: Output directory for saving models and logs (default: outputs)
batch_size: Per device training batch size (default: 1)
learning_rate: Learning rate (default: 2e-4)
fp16: Enable FP16 training (optional)
dataset: Dataset path or HuggingFace DataSet ID (default: Abirate/english_quotes)
use_local_dataset: Use a local dataset instead of HuggingFace DataSet (optional)
num_train_epochs: Number of training epochs (default: 1)
sample_key: Accessing the input samples from the dictionary using the specified key, applying the tokenizer on it (default: quote)
````

### Example Execution
````
python3 fine_tune_by_epoch.py --model_id pabligme/Llama_13b_chat --use_local_dataset --dataset /cnvrg/merged_final_ultimate_andy.json --num_train_epochs 1
````
### Notes
If you are using a local dataset, provide the appropriate sample key associated with the dataset. For example, use "input" as the sample key for our dataset example.

Feel free to reach out for any clarifications or assistance. Happy training!
