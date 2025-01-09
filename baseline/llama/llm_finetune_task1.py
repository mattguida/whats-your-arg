from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm
import pandas as pd
import jsonlines as jl
from sklearn.model_selection import StratifiedKFold

max_seq_length = 4000 
dtype = None 
load_in_4bit = False 

system_instruction = '''Analyze whether the following comment contains an argument about gay marriage.

Instructions:
1. Determine if the comment explicitly or implicitly uses the given argument
2. Assign a binary label:
   - 1 if the argument is present
   - 0 if the argument is not present
Requirements:
- Only use 1 or 0 as labels
- Do not repeat or include the input text in the response
- Focus solely on the presence/absence of the specific argument
'''

model, tokenizer = FastLanguageModel.from_pretrained(
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
max_seq_length = max_seq_length,
dtype = dtype,
load_in_4bit = load_in_4bit,
)    

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN


def formatting_prompts_func(example):
    # print(example)
    instruction = system_instruction
    input1 = example['text']
    input2 = example['argument']
    output = example['label']
    # print(input)
    prompt = alpaca_prompt.format(instruction, input1, input2, output) + EOS_TOKEN
    #print(prompt)
    return { "text" : prompt }


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Comment to analyze: {}
Argument to check for: {}

### Response:
{}"""

dataset = load_dataset("csv", data_files={"train": "data/fine_tuning/task1_finetune_data.csv"})

dataset = dataset.map(formatting_prompts_func)

def evaluate(eval_set, model, tokenizer):
    rows = []
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    to_write = []
    for sample in tqdm(eval_set):
        sample_input = sample['text'].split('### Response:')[0] + '### Response:'
        sample_label = sample['text'].split('### Response:')[1].strip()
        sample_label = sample_label.strip('<|end_of_text|>')
        sample_label = sample_label.strip('<|eot_i')
       
        inputs = tokenizer(sample_input, return_tensors = "pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True)
        text_output=tokenizer.batch_decode(outputs)[0]
        prediction = text_output.split('### Response:')[1]
        
        prediction = prediction.strip()
        prediction = prediction.strip('<|end_of_text|>')
        prediction = prediction.strip('<|eot_id|>')
        
        to_write.append({"label": sample_label, "prediction": prediction})

    return to_write


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

X = dataset["train"]["text"]  # Features (can be multiple columns if needed)
y = dataset["train"]["label"]  # Target labels
print(X[0])
print(y[0])

# Loop through the folds
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    train_dataset = dataset['train'].select(train_idx)
    eval_dataset = dataset['train'].select(test_idx)
    
    print(f"Fold {fold + 1}")
    print(f"Train size: {len(train_dataset)}, Test size: {len(eval_dataset)}")


    ds = train_dataset.train_test_split(test_size=0.2)
    print('Dataset')
    print(f"Test: {len(eval_dataset)}")
    print(f"Train:{len(ds['train'])}")
    print(f"Dev: {len(ds['test'])}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        )

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    print("Model loaded")

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",   
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
    )

    print("model prepared")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset = ds['train'],
        eval_dataset = ds['test'],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 100,
            max_steps = 4000,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs_task1_"+ str(fold)
        ),

    )
    print("trainer")

    trainer_stats = trainer.train()
    
    save_path = 'train_0.2_llama3_8b_lora_split_' + str(fold)
    model.save_pretrained(save_path) 
    
    tokenizer.save_pretrained(save_path)

    out_text= evaluate(eval_dataset, model, tokenizer)

    with jl.open('output_lora_task1_'+str(fold) + '.jsonl','w') as w:
        for item in out_text:
            w.write(item)
    w.close()

    del model
    del tokenizer
    torch.cuda.empty_cache()

