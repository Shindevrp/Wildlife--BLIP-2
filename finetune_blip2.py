from transformers import Blip2Processor, Blip2ForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset
from PIL import Image
import torch
import json

# Load processor and model
model_name = "Salesforce/blip2-flan-t5-xl"
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load your dataset
with open("data.json") as f:
    data = json.load(f)
dataset = Dataset.from_list(data)

# Preprocess function for captioning and VQA
def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    if "caption" in example:
        prompt = "Describe the image"
        label = example["caption"]
    elif "question" in example and "answer" in example:
        prompt = example["question"]
        label = example["answer"]
    else:
        raise ValueError("Example must have either 'caption' or both 'question' and 'answer'.")
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels = processor(text=label, return_tensors="pt", padding=True).input_ids
    inputs["labels"] = labels[0]
    return {key: val[0] for key, val in inputs.items()}

processed = dataset.map(preprocess)

# Training arguments
args = TrainingArguments(
    output_dir="./blip2-wildlife",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=processed,
    tokenizer=processor.tokenizer,
)

trainer.train()
