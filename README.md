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
