"""
Training Script for BERT Spam Classifier

This script fine-tunes the BERT model on the email spam detection task.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
from bert_spam_model import BERTSpamClassifier, get_tokenizer, preprocess_email


class EmailDataset(Dataset):
    """Dataset class for email spam detection."""
    
    def __init__(self, csv_file, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV file with email data
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.emails = []
        self.labels = []
        
        # Load data from CSV
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Combine subject and body for email content
                email_text = f"{row['subject']} {row['body']}"
                self.emails.append(email_text)
                self.labels.append(int(row['label_id']))
    
    def __len__(self):
        return len(self.emails)
    
    def __getitem__(self, idx):
        email_text = self.emails[idx]
        label = self.labels[idx]
        
        # Tokenize email
        encoding = self.tokenizer(
            email_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    predictions = []
    true_labels = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """Main training function."""
    # Configuration
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    MAX_LENGTH = 512
    MODEL_NAME = 'bert-base-uncased'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(MODEL_NAME)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = EmailDataset('data/train_emails.csv', tokenizer, MAX_LENGTH)
    test_dataset = EmailDataset('data/test_emails.csv', tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = BERTSpamClassifier(model_name=MODEL_NAME, num_labels=2, dropout_rate=0.3)
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(model, test_loader, device)
        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1-Score: {metrics['f1']:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/bert_spam_classifier.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': MODEL_NAME,
        'num_labels': 2,
        'dropout_rate': 0.3
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save tokenizer
    tokenizer.save_pretrained('models/tokenizer')
    print("Tokenizer saved to models/tokenizer")


if __name__ == '__main__':
    main()

