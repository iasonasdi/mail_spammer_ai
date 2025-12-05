"""
Training Data Generation Script

This script generates synthetic email data for training the BERT spam classifier.
The data is saved in the 'data' folder.
"""

import os
import csv
import random
from datetime import datetime, timedelta


def generate_ham_emails():
    """Generate legitimate (ham) email examples."""
    ham_templates = [
        {
            'subject': 'Meeting scheduled for next week',
            'body': 'Hi, I would like to schedule a meeting for next week to discuss the project progress. Please let me know your availability.',
            'sender': 'colleague@company.com'
        },
        {
            'subject': 'Project update',
            'body': 'Hello team, I wanted to provide an update on the current project status. We are on track with the deliverables.',
            'sender': 'manager@company.com'
        },
        {
            'subject': 'Invoice #{}'.format(random.randint(1000, 9999)),
            'body': 'Dear customer, please find attached the invoice for your recent purchase. Payment is due within 30 days.',
            'sender': 'billing@company.com'
        },
        {
            'subject': 'Thank you for your application',
            'body': 'Thank you for applying to our position. We have received your resume and will review it shortly.',
            'sender': 'hr@company.com'
        },
        {
            'subject': 'Weekly report',
            'body': 'Please find attached the weekly sales report. Let me know if you have any questions.',
            'sender': 'sales@company.com'
        },
        {
            'subject': 'Reminder: Team lunch tomorrow',
            'body': 'This is a reminder that we have a team lunch scheduled for tomorrow at 12:30 PM. See you there!',
            'sender': 'team@company.com'
        },
        {
            'subject': 'Document review request',
            'body': 'Could you please review the attached document and provide your feedback by Friday?',
            'sender': 'reviewer@company.com'
        },
        {
            'subject': 'Password reset confirmation',
            'body': 'Your password has been successfully reset. If you did not request this change, please contact support immediately.',
            'sender': 'security@company.com'
        },
        {
            'subject': 'Welcome to our service',
            'body': 'Welcome! We are excited to have you on board. Here are some resources to help you get started.',
            'sender': 'support@company.com'
        },
        {
            'subject': 'Quarterly review meeting',
            'body': 'The quarterly review meeting has been scheduled for next month. Please prepare your reports.',
            'sender': 'admin@company.com'
        }
    ]
    
    ham_emails = []
    for i in range(500):  # Generate 500 ham emails
        template = random.choice(ham_templates)
        email = {
            'id': f'ham_{i+1}',
            'subject': template['subject'],
            'body': template['body'],
            'sender': template['sender'],
            'label': 'ham',
            'label_id': 0
        }
        ham_emails.append(email)
    
    return ham_emails


def generate_spam_emails():
    """Generate spam email examples."""
    spam_templates = [
        {
            'subject': 'URGENT: Claim your prize now!!!',
            'body': 'Congratulations! You have won $1,000,000! Click here to claim your prize immediately. Limited time offer!',
            'sender': 'winner@prize.com'
        },
        {
            'subject': 'Act now - 90% OFF!!!',
            'body': 'Exclusive offer just for you! Get 90% off on all products. Click now before this offer expires!',
            'sender': 'deals@shopping.com'
        },
        {
            'subject': 'Your account will be closed',
            'body': 'URGENT: Your account will be closed in 24 hours unless you verify your information. Click here to verify now!',
            'sender': 'security@bank.com'
        },
        {
            'subject': 'Make money fast - work from home',
            'body': 'Earn $5000 per week working from home! No experience needed. Start making money today!',
            'sender': 'jobs@workfromhome.com'
        },
        {
            'subject': 'FREE!!! Click to claim',
            'body': 'You have been selected for a free gift! Click here to claim your free iPhone, laptop, or cash prize!',
            'sender': 'free@prizes.com'
        },
        {
            'subject': 'Viagra - Special offer',
            'body': 'Buy cheap medications online. Special discount on all products. Order now and save big!',
            'sender': 'pharmacy@meds.com'
        },
        {
            'subject': 'Nigerian prince needs your help',
            'body': 'I am a prince and need your help to transfer $10 million. I will give you 20% if you help me. Reply urgently!',
            'sender': 'prince@nigeria.com'
        },
        {
            'subject': 'Your password expired - verify now',
            'body': 'Your password has expired. Click here to verify your account or it will be permanently deleted.',
            'sender': 'noreply@service.com'
        },
        {
            'subject': 'You have been selected!!!',
            'body': 'Congratulations! You have been randomly selected to win a luxury vacation. Click here to claim!',
            'sender': 'contest@winner.com'
        },
        {
            'subject': 'Investment opportunity - guaranteed returns',
            'body': 'Exclusive investment opportunity with guaranteed 500% returns in 30 days. Invest now and become rich!',
            'sender': 'invest@opportunity.com'
        }
    ]
    
    spam_emails = []
    for i in range(500):  # Generate 500 spam emails
        template = random.choice(spam_templates)
        email = {
            'id': f'spam_{i+1}',
            'subject': template['subject'],
            'body': template['body'],
            'sender': template['sender'],
            'label': 'spam',
            'label_id': 1
        }
        spam_emails.append(email)
    
    return spam_emails


def save_to_csv(emails, filename):
    """Save emails to a CSV file."""
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'subject', 'body', 'sender', 'label', 'label_id'])
        writer.writeheader()
        writer.writerows(emails)
    
    print(f"Saved {len(emails)} emails to {filepath}")


def main():
    """Generate and save training data."""
    print("Generating training data...")
    
    # Generate ham and spam emails
    ham_emails = generate_ham_emails()
    spam_emails = generate_spam_emails()
    
    # Combine and shuffle
    all_emails = ham_emails + spam_emails
    random.shuffle(all_emails)
    
    # Split into train (80%) and test (20%)
    split_idx = int(len(all_emails) * 0.8)
    train_emails = all_emails[:split_idx]
    test_emails = all_emails[split_idx:]
    
    # Save to CSV files
    save_to_csv(train_emails, 'train_emails.csv')
    save_to_csv(test_emails, 'test_emails.csv')
    
    print(f"\nData generation complete!")
    print(f"Total emails: {len(all_emails)}")
    print(f"Training set: {len(train_emails)} emails")
    print(f"Test set: {len(test_emails)} emails")
    print(f"Ham emails: {len(ham_emails)}")
    print(f"Spam emails: {len(spam_emails)}")


if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    main()

