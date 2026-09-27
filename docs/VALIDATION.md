# Validation record

Review date: 27 September 2026. The 5 code cells in the included notebook were executed in order
in a fresh IPython process launched from `notebooks/`, on Linux with Python 3.12 and CPU execution.
The saved notebook contains the resulting text and figure outputs, with no saved execution errors.

Three regression tests passed: invalid labels/nonfinite features are rejected; training batch sizes
avoid one-row BatchNorm batches; checkpoint inference preserves feature order and scaler statistics.
The notebook trained on 120 rows, selected epoch 46 using 40 validation rows, and evaluated 40 test rows.
It explicitly checked the restored validation loss and equality of test probabilities after checkpoint reload.

Core package versions match the pins in `requirements.txt`. The notebook web interface and installation
on Windows/macOS were not separately exercised. Stochastic results can vary across platforms and
library builds. This validation covers the supplied example and focused regression cases, not every
possible input or production deployment.

Re-run regression checks from the repository root:

```bash
python -m unittest discover -s tests -v
```
