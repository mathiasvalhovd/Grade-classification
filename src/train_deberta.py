"""Fine‑tune DeBERTa‑v3‑small on cleaned JSONL data.

Usage:
    python train_deberta.py --train data/processed/train.jsonl ...
"""

import argparse, json, os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/processed/train.jsonl')
    parser.add_argument('--val', default='data/processed/val.jsonl')
    parser.add_argument('--model_name', default='microsoft/deberta-v3-small')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    train_ds = load_dataset('json', data_files=args.train, split='train')
    val_ds   = load_dataset('json', data_files=args.val, split='train')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tok(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)
    train_ds = train_ds.map(tok, batched=True)
    val_ds   = val_ds.map(tok, batched=True)

    train_ds = train_ds.rename_column('target', 'labels')
    val_ds   = val_ds.rename_column('target', 'labels')

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir='models/deberta',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy='epoch',
        fp16=args.fp16,
        save_strategy='epoch',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    trainer.save_model('models/deberta/best')

if __name__ == '__main__':
    main()
