import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMClassifier(nn.Module):
    """Hybrid CNN + BiLSTM for text classification."""

    def __init__(self, vocab_size, embed_dim=300, num_classes=2, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, k) for k in (3,4,5)
        ])
        self.lstm = nn.LSTM(input_size=128*3, hidden_size=256,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256*2, num_classes)

    def forward(self, x):
        # x: (B, T)
        emb = self.embed(x).transpose(1,2)  # (B, D, T)
        c = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        c = torch.cat(c, dim=1).unsqueeze(1)  # (B, 1, 128*3)
        out, _ = self.lstm(c)
        logits = self.fc(out[:, -1, :])
        return logits
