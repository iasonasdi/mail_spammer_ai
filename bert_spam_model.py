"""
BERT-based Email Spam Detection Model

This module implements a BERT model modified for binary classification (spam/ham).
The key modifications to the base BERT model are:
1. Classification head: A linear layer added on top of BERT's [CLS] token
2. Dropout for regularization
3. Binary classification output (spam vs ham)
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig


class BERTSpamClassifier(nn.Module):
    """
    BERT-based spam detection model.
    
    Architecture:
    - Base: Pre-trained BERT model (bert-base-uncased)
    - Modification 1: Extract [CLS] token embedding (sentence representation)
    - Modification 2: Add dropout layer for regularization
    - Modification 3: Add linear classification head (768 -> 1 for binary classification)
    """
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2, dropout_rate=0.3):
        """
        Initialize the BERT spam classifier.
        
        Args:
            model_name: Pre-trained BERT model name
            num_labels: Number of classification labels (2 for spam/ham)
            dropout_rate: Dropout probability for regularization
        """
        super(BERTSpamClassifier, self).__init__()
        
        # Load pre-trained BERT model (frozen initially, will be fine-tuned)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Get the hidden size from BERT config (typically 768 for bert-base)
        self.hidden_size = self.bert.config.hidden_size
        
        # Modification 1: Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Modification 2: Classification head
        # Takes BERT's 768-dimensional [CLS] token embedding and outputs logits
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Mask to avoid attention on padding tokens
            token_type_ids: Segment embeddings (not used for single sentences)
            
        Returns:
            logits: Raw classification scores (before softmax)
        """
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Extract [CLS] token embedding (first token, index 0)
        # This token contains the aggregated sentence representation
        pooled_output = outputs.pooler_output  # Shape: (batch_size, hidden_size)
        
        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        # Pass through classification head
        logits = self.classifier(pooled_output)  # Shape: (batch_size, num_labels)
        
        return logits


def get_tokenizer(model_name='bert-base-uncased'):
    """
    Get the BERT tokenizer for preprocessing text.
    
    Args:
        model_name: Pre-trained BERT model name
        
    Returns:
        tokenizer: BERT tokenizer instance
    """
    return BertTokenizer.from_pretrained(model_name)


def preprocess_email(email_text, tokenizer, max_length=512):
    """
    Preprocess email text for BERT input.
    
    BERT expects:
    - Tokenized text with special tokens [CLS] and [SEP]
    - Attention mask to handle variable-length sequences
    - Padded/truncated to max_length
    
    Args:
        email_text: Raw email text
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length (BERT's limit is 512)
        
    Returns:
        Dictionary with input_ids, attention_mask, and token_type_ids
    """
    # Tokenize and encode the email text
    encoding = tokenizer(
        email_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'token_type_ids': encoding.get('token_type_ids', None)
    }

