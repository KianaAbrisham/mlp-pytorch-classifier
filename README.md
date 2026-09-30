# PyTorch MLP for Binary Tabular Classification

[![Checks](https://github.com/KianaAbrisham/mlp-pytorch-classifier/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/KianaAbrisham/mlp-pytorch-classifier/actions/workflows/checks.yml)

Train a small multilayer perceptron on numeric CSV features, select its checkpoint using
validation loss, and reload the model with the same preprocessing for inference.
The included **200-row synthetic dataset** demonstrates the workflow; it is not a real-world benchmark.

## Workflow

- Stratified 60/20/20 train/validation/test split with a fixed seed.
- Standardization fitted on training rows only.
- Two hidden layers (64 and 32 units), BatchNorm, ReLU and dropout.
- Adam optimization and early stopping based on validation loss.
- Fresh test predictions after restoring the selected checkpoint.
- A saved checkpoint containing weights, scaler statistics, feature order, split membership and selected epoch.

The executed demo selected epoch 46 and reached 0.90 accuracy on 40 synthetic test rows.
That small result shows execution of the example, not expected performance on another dataset.

## Files and data

| Path | Purpose |
|---|---|
| [notebooks/mlp_tabular.ipynb](notebooks/mlp_tabular.ipynb) | Executed training, evaluation and reload example |
| [src/model.py](src/model.py) | MLP and checkpoint inference |
| [src/data_utils.py](src/data_utils.py) | CSV validation and training batch-size helper |
| [data/sample.csv](data/sample.csv) | Synthetic features `f1`–`f10` and binary `label` |
| [tests/test_mlp.py](tests/test_mlp.py) | Invalid-input, BatchNorm and checkpoint regression checks |

For another CSV, edit the path and feature list in the notebook. All selected features must
be finite numeric values; labels must be exactly 0 or 1. IDs should not be predictors.
The split assumes independent rows; repeated subjects or time series need an appropriate
group or time split. Very small class counts may not support stratification.

Running the notebook creates `outputs/mlp_demo.pt` and `outputs/training_history.csv`.
Rerunning overwrites these demo files. The checkpoint is generated locally and is not included in Git.

## Run locally

Use Python 3.12 and a separate environment for this project. From the repository folder:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` in Windows Command Prompt or
`source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter notebook notebooks/mlp_tabular.ipynb
```

The notebook finds the repository from either its root folder or `notebooks/`.
The saved outputs come from CPU execution with the included data; see
[validation](docs/VALIDATION.md) for the checks and limits.

[Development notes](https://github.com/KianaAbrisham/KianaAbrisham/blob/main/docs/DEVELOPMENT.md)

## License

MIT — see [LICENSE](LICENSE).
