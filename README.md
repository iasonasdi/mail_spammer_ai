# BERT Email Spam Detection

A BERT-based AI model for detecting spam emails. This project implements a fine-tuned BERT model with modifications for binary classification (spam vs. ham).

## Project Structure

```
mail_spammer_ai/
├── data/                          # Generated training data folder
│   ├── train_emails.csv          # Training dataset
│   └── test_emails.csv           # Test dataset
├── models/                        # Saved models folder
│   ├── bert_spam_classifier.pt   # Trained model weights
│   └── tokenizer/                # Saved tokenizer
├── bert_spam_model.py            # BERT model implementation
├── generate_training_data.py     # Script to generate training data
├── train_model.py                # Training script
├── predict.py                    # Inference/prediction script
├── requirements.txt              # Python dependencies
├── BERT_MODIFICATIONS.md         # Detailed explanation of BERT modifications
└── README.md                     # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python generate_training_data.py
```

This will create:
- `data/train_emails.csv` - 800 training emails (400 ham, 400 spam)
- `data/test_emails.csv` - 200 test emails (100 ham, 100 spam)

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Load the pre-trained BERT model (bert-base-uncased)
- Fine-tune it on the email spam detection task
- Save the trained model to `models/bert_spam_classifier.pt`
- Display training progress and evaluation metrics

### 4. Make Predictions

```bash
python predict.py
```

This will load the trained model and make predictions on example emails.

## Understanding the BERT Modifications

The base BERT model is modified in three key ways for spam detection:

### 1. Classification Head
- A linear layer (768 → 2) added on top of BERT
- Converts BERT's sentence embeddings to spam/ham predictions

### 2. [CLS] Token Usage
- Uses the special [CLS] token embedding as the sentence representation
- This token aggregates information about the entire email

### 3. Dropout Regularization
- Added dropout layer (30%) to prevent overfitting
- Helps the model generalize to new emails

For detailed explanations, see [BERT_MODIFICATIONS.md](BERT_MODIFICATIONS.md).

## How It Works

1. **Input**: Email text (subject + body)
2. **Tokenization**: BERT tokenizer converts text to tokens
3. **BERT Processing**: 12 transformer layers process the tokens
4. **Classification**: [CLS] token → Dropout → Linear layer → Prediction
5. **Output**: Spam (1) or Ham (0) with confidence scores

## Model Architecture

```
Email Text
    ↓
BERT Tokenizer
    ↓
BERT Encoder (12 layers)
    ↓
[CLS] Token (768 dims)
    ↓
Dropout (0.3)
    ↓
Linear Classifier (768 → 2)
    ↓
Spam/Ham Prediction
```

## Training Details

- **Base Model**: bert-base-uncased (110M parameters)
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Max Sequence Length**: 512 tokens
- **Optimizer**: AdamW with linear warmup scheduler

## Usage Example

```python
from bert_spam_model import BERTSpamClassifier, get_tokenizer
from predict import load_model, predict_spam

# Load model
model, device = load_model('models/bert_spam_classifier.pt')
tokenizer = get_tokenizer('models/tokenizer')

# Predict
email = "URGENT: Claim your prize now!!!"
result = predict_spam(email, model, tokenizer, device)
print(f"Prediction: {result['label']} ({result['confidence']:.2%})")
```

## Data Format

The training data CSV files have the following columns:
- `id`: Unique email identifier
- `subject`: Email subject line
- `body`: Email body text
- `sender`: Email sender address
- `label`: 'ham' or 'spam'
- `label_id`: 0 (ham) or 1 (spam)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- scikit-learn 1.3+
- numpy 1.24+

## Notes

- The model downloads BERT weights on first run (~440MB)
- Training takes ~10-30 minutes on GPU, longer on CPU
- Generated data is synthetic; for production, use real email datasets
- Model accuracy depends on training data quality and diversity

## License

This project is for educational purposes.
