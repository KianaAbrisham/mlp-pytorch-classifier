# PyTorch MLP Classifier (Tabular) — Portfolio Sample

This repository shows a **multilayer perceptron (MLP)** in **PyTorch** for binary classification on tabular data.
It uses a clean training loop (with early stopping), metric tracking, and neat plots (loss & accuracy).

## What this demonstrates
- PyTorch **Dataset/DataLoader** for CSV tabular data
- MLP with **batch norm** and **dropout**
- Training loop with **early stopping** and model checkpointing
- Evaluation: accuracy, precision/recall/F1, confusion matrix
- Reproducible notebook with clear narrative

## Structure
```
.
├── notebooks
│   └── mlp_tabular.ipynb          # End-to-end training & eval with plots
├── src
│   └── data_utils.py              # CSVDataset + helpers
│   └── model.py                   # Simple MLP
├── data
│   └── sample.csv                 # Tiny demo CSV (features f1..f10, label)
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/mlp_tabular.ipynb
```

### Use your own data
- Provide a CSV with numeric feature columns and a `label` column (0/1).
- Update the `csv_path` cell in the notebook and run all cells.

## License
MIT
