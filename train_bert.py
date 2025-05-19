import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import os

# Enable mixed precision
from torch.amp import GradScaler, autocast

# Custom Dataset for Fake News
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to load and preprocess data
def load_and_preprocess_data():
    try:
        true_df = pd.read_csv("true.csv")
        fake_df = pd.read_csv("fake.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure true.csv and fake.csv are in D:\\fake")
        exit(1)

    # Assign labels
    true_df['label'] = 0  # Real news
    fake_df['label'] = 1  # Fake news

    # Check for 'text' column
    if 'text' not in true_df.columns or 'text' not in fake_df.columns:
        print("Error: 'text' column not found.")
        print("true.csv columns:", true_df.columns.tolist())
        print("fake.csv columns:", fake_df.columns.tolist())
        print("Update the column name in the script (e.g., 'content' instead of 'text').")
        exit(1)

    # Select 'text' and 'label'
    true_df = true_df[['text', 'label']]
    fake_df = fake_df[['text', 'label']]

    # Combine datasets
    df = pd.concat([true_df, fake_df], ignore_index=True)

    # Handle missing values
    df['text'] = df['text'].fillna('')

    # Split data
    X = df['text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Main training function
def train_model():
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=2)

    # Create datasets
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer)
    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, pin_memory=True)

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: CUDA not available. Training on CPU, which will be slower.")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler('cuda')

    # Training loop
    epochs = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Mixed precision training
            with autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with autocast('cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch + 1}, Test Accuracy: {accuracy:.4f}")

    # Save model and tokenizer
    model.save_pretrained("fake_news_bert_model")
    tokenizer.save_pretrained("fake_news_bert_model")
    print("Model and tokenizer saved to fake_news_bert_model")

if __name__ == '__main__':
    train_model()