The Finetune Repo was created to test benchmarking while finetuning on cnvrg.io.
before you start remember we are training on GPU, if you want to train on cpu you must change this: device_map={"":0} to this: device_map={"":"cpu"}.
the model by default is using the meta-llama/Llama-2-7b-chat-hf model along with Abirate/english_quotes Dataset, you can change those as you wish,
also in fine_tune_by_epoch.py I added the option to use local datasets as well if passing this arg: --use_local_dataset and then giving it the path to the dataset

before you start a job you need to know that you must pass your HuggingFace token as an environment variable under the project settings like so:

HF_TOKEN = { YOUR_HUGGINGFACE_TOKEN }
** note that for Meta Llama2 you will need an approved HF token, for that please ping me in slack :-)

the fine_tune_by_steps.py code can be executed as is like so:
python3 fine_tune_by_steps.py 

as this will execute the default values of the args, or you can specify them while execution, this is the args list:

--model_id
--output_dir
--batch_size
--max_steps
--learning_rate
--fp16
--dataset

the code can be executed as follow:
python3 fine_tune_by_epoch.py --model_id pabligme/Llama_13b_chat --use_local_dataset --dataset /cnvrg/merged_final_ultimate_andy.json --num_train_epochs 1 

the code also can be executed by:
python3 fine_tune_by_epoch.py and this will take the default params
this is the args list for this script:

--model_id
--output_dir
--batch_size
--num_train_epochs
--learning_rate
--fp16
--dataset
--use_local_dataset
--sample_key

if you are using the default dataset the default sample will be quote, but if you intend to use a local dataset please use a sample key that is what associate with the dataset
in our dataset example you must use: "input" as sample key


