# roberta_text_classifier_manual.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW 
import pandas as pd

# 1. Dummy dataset
data = {
    'text': [
        "This essay explains economic growth over decades.",  # Human
        "As an AI language model, I believe that the economy is driven by...",  # AI
    ],
    'label': [0, 1]  # 0 = Human, 1 = AI
}
df = pd.DataFrame(data)

# 2. Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# 3. Custom Dataset
class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx])
        }
        return item

# 4. DataLoader
dataset = TextDataset(df)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. Model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 6. Training Setup
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# 7. Training Loop
model.train()
for epoch in range(3):
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")

# 8. Prediction Function
def predict(text):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        input
