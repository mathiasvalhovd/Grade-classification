# 🎓 Student-Performance Classification  
*ML Final Project – Tabular Deep-Learning Models*

We will predict **student achievement** from socio-academic factors using the  
[Kaggle *Student Performance Factors* dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors).  
Instead of forecasting a raw test score we convert the numeric **`Exam_Score`** into **three classes**:

| Class | Rule (default) | Meaning |
|-------|----------------|---------|
| **Bad**    | `score < 60`      | Fail / poor performance |
| **Medium** | `60 ≤ score < 80` | Average |
| **Good**   | `score ≥ 80`      | High achievement |

*Thresholds live in `config.yaml`; adjust to your grading scheme or switch to equal-frequency (tercile) binning with one flag.*

---

## 📁 Folder Layout

```text
student_performance/
├── data/
│   ├── raw/          # ⬇️  put student-performance-factors.csv here
│   └── processed/    # train/val/test CSVs after cleaning + target bins
├── notebooks/        # EDA & preprocessing notebooks
├── src/              # training / evaluation scripts
├── models/           # checkpoints & logs
├── requirements.txt  # Python deps
└── README.md         # this file
```

## 🛠️ Step-by-Step: Set Up & Use a Python Virtual Environment

This project uses a Python virtual environment and `requirements.txt`  
for reproducible, isolated installation of all dependencies.

---

### 1. **Create the virtual environment**

Open a terminal in your project root and run:

```bash
python -m venv .venv
```
## 2. Activate the virtual environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

## 3. Upgrade pip (recommended)

Upgrade pip to the latest version by running:

```bash
pip install --upgrade pip
```

## 4. Install dependencies from requirements.txt

Make sure your `requirements.txt` file is in the root folder, then run:

```bash
pip install -r requirements.txt
```