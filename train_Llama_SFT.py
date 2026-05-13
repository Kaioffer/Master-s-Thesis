from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from trl import SFTTrainer
from trl import SFTConfig
from datasets import load_dataset

# 1. Configuration
max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True
random_seed = 42
dataset_file = "rover_ltl_train.jsonl"

# Save artifacts outside OneDrive workspace.
documents_dir = Path.home() / "Documents"
training_artifacts_dir = documents_dir / "llama_training"
output_dir = str(training_artifacts_dir / "outputs")
model_output_dir = str(training_artifacts_dir / "llama3_ltl_specialist")
training_artifacts_dir.mkdir(parents=True, exist_ok=True)

# 2. Load Llama 3 8B
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8b-instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 3. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 4. Load and split JSONL data
raw_dataset = load_dataset("json", data_files=dataset_file, split="train")
raw_dataset = raw_dataset.shuffle(seed=random_seed)
split = raw_dataset.train_test_split(test_size=0.05, seed=random_seed)
train_dataset = split["train"]
eval_dataset = split["test"]

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def format_prompts(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Create a list of messages for the native tokenizer
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts }

train_dataset = train_dataset.map(format_prompts, batched=True)
eval_dataset = eval_dataset.map(format_prompts, batched=True)

# 5. Set up trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=4,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=random_seed,
        output_dir=output_dir,
        padding_free=False,
    ),
)

# 6. Train
trainer_stats = trainer.train()
print(trainer_stats)

# 7. Save final domain model
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)