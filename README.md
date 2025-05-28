# Phishing Email Detection – ML Final Project

**Models:**  
1. Fine‑tuned *DeBERTa‑v3‑small* (Transformer)  
2. Hybrid *CNN + Bi‑LSTM* baseline

## Quickstart

```bash
git clone <your‑repo‑url> phishing_project
cd phishing_project
pip install -r requirements.txt
```

1. Download the Kaggle CSV and place it in `data/raw/`.
2. Run the notebook in `notebooks/phish_cleaning.ipynb` to create cleaned splits and a vocab.
3. Fine‑tune DeBERTa:

```bash
python src/train_deberta.py --fp16
```

4. Train CNN‑BiLSTM:

```bash
python src/train_cnn_lstm.py
```

## Folder structure

```
phishing_project/
  ├── data/
  │   ├── raw/          ← original CSV
  │   └── processed/    ← train/val/test JSONL + vocab
  ├── notebooks/        ← Jupyter notebooks
  ├── src/              ← training scripts & model code
  ├── models/           ← saved checkpoints
  ├── README.md
  ├── requirements.txt
  └── .gitignore
```
