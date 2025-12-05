"""
Inference Script for BERT Spam Classifier

This script loads a trained model and makes predictions on new emails.
"""

import torch
import torch.nn as nn
from bert_spam_model import BERTSpamClassifier, get_tokenizer, preprocess_email


def load_model(model_path, model_name='bert-base-uncased', num_labels=2, dropout_rate=0.3):
    """Load a trained model from disk."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = BERTSpamClassifier(model_name=model_name, num_labels=num_labels, dropout_rate=dropout_rate)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def predict_spam(email_text, model, tokenizer, device, max_length=512):
    """
    Predict if an email is spam or ham.
    
    Args:
        email_text: The email text to classify
        model: Trained BERT model
        tokenizer: BERT tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with prediction, probability, and confidence
    """
    # Preprocess email
    encoding = tokenizer(
        email_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    label = 'spam' if prediction == 1 else 'ham'
    
    return {
        'label': label,
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': {
            'ham': probabilities[0][0].item(),
            'spam': probabilities[0][1].item()
        }
    }


def main():
    """Example usage of the prediction function."""
    # Load model and tokenizer
    print("Loading model...")
    model_path = 'models/bert_spam_classifier.pt'
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model_name = checkpoint.get('model_name', 'bert-base-uncased')
        num_labels = checkpoint.get('num_labels', 2)
        dropout_rate = checkpoint.get('dropout_rate', 0.3)
    except:
        model_name = 'bert-base-uncased'
        num_labels = 2
        dropout_rate = 0.3
    
    model, device = load_model(model_path, model_name, num_labels, dropout_rate)
    
    # Try to load saved tokenizer, otherwise use default
    try:
        tokenizer = get_tokenizer('models/tokenizer')
    except:
        tokenizer = get_tokenizer(model_name)
    
    print("Model loaded successfully!\n")
    
    # Example emails to test
    test_emails = [
        {
            'subject': 'Meeting scheduled for next week',
            'body': 'Hi, I would like to schedule a meeting for next week to discuss the project progress.'
        },
        {
            'subject': 'URGENT: Claim your prize now!!!',
            'body': 'Congratulations! You have won $1,000,000! Click here to claim your prize immediately.'
        },
        {
            'subject': 'Project update',
            'body': 'Hello team, I wanted to provide an update on the current project status.'
        },
        {
            'subject': 'Act now - 90% OFF!!!',
            'body': 'Exclusive offer just for you! Get 90% off on all products. Click now!'
        }
    ]
    
    # Make predictions
    for email in test_emails:
        email_text = f"{email['subject']} {email['body']}"
        result = predict_spam(email_text, model, tokenizer, device)
        
        print(f"Subject: {email['subject']}")
        print(f"Prediction: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities - Ham: {result['probabilities']['ham']:.2%}, Spam: {result['probabilities']['spam']:.2%}")
        print("-" * 50)


if __name__ == '__main__':
    main()

