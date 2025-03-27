import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

import torch
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='')
    
# Adding arguments
parser.add_argument('--model_name', type=str)
parser.add_argument('--experiment', type=str)
args = parser.parse_args()

model_name = args.model_name
experiment = args.experiment


clarity_mapping ={
    '1.1 Explicit': 'Direct Reply',
    '1.2 Implicit': 'Indirect',
    '2.1 Dodging': "Indirect",
    '2.2 Deflection': "Indirect",
    '2.3 Partial/half-answer': "Indirect",
    '2.4 General': "Indirect",
    '2.6 Declining to answer': "Direct Non-Reply",
    '2.7 Claims ignorance': "Direct Non-Reply",
    '2.8 Clarification': "Direct Non-Reply",
}

if experiment == "evasion_based_clarity":
    num_labels = 11
    mapping_labels = {'1.1 Explicit': 0, '1.2 Implicit': 1, '2.1 Dodging': 2, '2.2 Deflection': 3, '2.3 Partial/half-answer': 4, '2.4 General': 5, '2.5 Contradictory': 6, '2.6 Declining to answer': 7, '2.7 Claims ignorance': 8, '2.8 Clarification': 9, '2.9 Diffusion': 10}
elif experiment == "direct_clarity":
    num_labels = 3
    mapping_labels = {"Direct Reply": 0, "Indirect": 1, "Direct Non-Reply": 2}


# Load pre-trained RoBERTa model and tokenizer
if 'roberta' in model_name: 
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("cuda")
    max_size = 512

elif "xlnet" in model_name:
    # Load pre-trained XLNet model and tokenizer
    from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("cuda")
    max_size = 4096

elif "deberta" in model_name: 
    from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
    model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("cuda")
    max_size = 512


# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, max_length=512):  # You can set max_length to an appropriate value

        self.max_length = max_length
        self.texts, self.labels = [], []
        for text, label in zip(texts, labels):
            inputs = tokenizer(  text,
                                return_tensors='pt',
                                # truncation=True,
                                padding='max_length',  # Use padding to ensure all sequences have the same length
                                max_length=self.max_length
                            )

            # Check if the input has more tokens than the max_length
            if len(inputs['input_ids'][0]) > self.max_length:
                continue
            self.texts.append(text)
            self.labels.append(label)

        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx],
            return_tensors='pt',
            # truncation=True,
            padding='max_length',  # Use padding to ensure all sequences have the same length
            max_length=self.max_length
        )

        # Check if the input has more tokens than the max_length
        if len(inputs['input_ids'][0]) > self.max_length:
            # If so, return None for inputs and label
            return None, None

        label = torch.tensor(self.labels[idx])
        return inputs, label


# Example data

dataset = pd.read_csv("dataset/QAEvasion.csv")


all_texts = [f"Question: {row['interview_question']}\n\nAnswer: {row['interview_answer']}\n\nSubanswer: {row['question']}" for row in dataset["train"] if "other" not in row["label"].lower()]

if experiment == "evasion_based_clarity":
    all_labels = [mapping_labels[row["label"]] for row in dataset["train"] if "other" not in row["label"].lower()]
elif experiment == "direct_clarity":
    all_labels = [mapping_labels[clarity_mapping[row["label"]]] for row in dataset["train"] if "other" not in row["label"].lower()]

print (set(all_labels))
print (len(all_texts))

train_texts, val_texts, train_labels, val_labels = all_texts[:2700], all_texts[2700:], all_labels[:2700], all_labels[2700:]

# Create datasets and dataloaders
train_dataset = CustomDataset(train_texts, train_labels, max_length=512)
val_dataset = CustomDataset(val_texts, val_labels, max_length=512)


def collate_fn(batch):
    inputs, labels = zip(*batch)
    return {
        'input_ids': torch.stack([x['input_ids'].squeeze() for x in inputs]),
        'attention_mask': torch.stack([x['attention_mask'].squeeze() for x in inputs]),
        'labels': torch.tensor(labels)
    }

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

print (len(train_dataloader), print (val_dataloader))

# Fine-tuning
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 5
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/ {num_epochs} - Training'):
        inputs = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        labels = batch['labels'].to("cuda")
    
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Inside the validation loop
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
            inputs = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")
    
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    
            # Calculate accuracy
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
    
    average_val_loss = val_loss / len(val_dataloader)
    accuracy = correct_preds / total_preds
    print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {average_val_loss:.4f} - Accuracy: {accuracy * 100:.2f}%')

# Save the fine-tuned model

out_file = f"{model_name.split('/')[-1]}-qaevasion-{experiment}"
model.save_pretrained(out_file)
