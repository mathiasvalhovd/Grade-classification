"""Train CNN‑BiLSTM on JSONL data using the vocab generated in notebooks."""
import argparse, json, os, re
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from model_cnn_lstm import CNNLSTMClassifier

def basic_tokenize(text):
    return re.findall(r"\b\w[\w'-]*\b", text)

class JsonlDataset(Dataset):
    def __init__(self, path, vocab, max_len=256):
        self.samples = [json.loads(l) for l in open(path, encoding='utf8')]
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        text = self.samples[idx]['text']
        label = self.samples[idx]['target']
        tokens = [self.vocab.get(tok, self.vocab['<unk>']) for tok in basic_tokenize(text)][:self.max_len]
        ids = tokens + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        return torch.tensor(ids), torch.tensor(label)

def load_vocab(path):
    return {tok.strip(): i for i, tok in enumerate(open(path, encoding='utf8'))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/processed/train.jsonl')
    parser.add_argument('--val', default='data/processed/val.jsonl')
    parser.add_argument('--vocab', default='data/processed/vocab.txt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    train_ds = JsonlDataset(args.train, vocab)
    val_ds   = JsonlDataset(args.val, vocab)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMClassifier(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        for xb, yb in tqdm(train_dl, desc=f'train {epoch}'):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        # simple val accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        acc = correct / total
        print(f'Epoch {epoch}: val accuracy={acc:.3f}')
    Path('models/cnn_lstm').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'models/cnn_lstm/model.pt')

if __name__ == '__main__':
    main()
