# 🛡️ Phishing-Email Detection (ML Final Project)

Compare a **state-of-the-art Transformer** (DeBERTa-v3-small) against a compact **CNN + Bi-LSTM** baseline on a public phishing-e-mail dataset.  
Everything you need—folder layout, environment setup, data-cleaning recipe, and training commands—is documented **in this single README**.

---

## 📌 Project Goals

| Goal | Why it matters |
|------|----------------|
| **Binary classification** (phish vs legit) | Core to enterprise e-mail security |
| **Model comparison** (SOTA vs lightweight) | Explore accuracy ↔ compute trade-offs |
| **Rapid delivery (≈ 2 weeks)** | Fits typical course / hackathon deadlines |

---

## 📁 Recommended Folder Structure

```text
phishing_project/
├── data/
│   ├── raw/          # drop the original Kaggle CSV here
│   └── processed/    # JSONL splits & vocab (after cleaning)
├── notebooks/        # Jupyter notebooks you create
├── src/              # (optional) training scripts
├── models/           # saved checkpoints (populated after training)
├── requirements.txt  # Python deps
└── README.md         # this file
```



## setup venv
python -m venv .venv
# macOS / Linux
source .venv/bin/activate


pip install --upgrade pip
pip install -r requirements.txt
# For GPU: 
follow the CUDA wheel instructions at https://pytorch.org

# verify installation
```
python - <<'PY'
import torch, transformers, pandas; print(
  "Torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
  "\nTransformers", transformers.__version__)
PY
```