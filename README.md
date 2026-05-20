# 2D Multi-View CNN Interpretability Framework

Wood NIR spectra framework for studying which wavelengths, preprocessing views, wavelength bands, and moisture/structure states a 2D CNN uses. The focus is physical interpretation rather than leaderboard accuracy.

## Data

Place `train.csv` and `test.csv` in `data/`. Both CSV files are read as `cp932`. Wavenumber columns are detected from numeric column names and converted to wavelength by:

```python
wavelengths = 1e7 / wavenumbers
```

The spectra are then sorted by ascending wavelength.

## Spectral Views

The model input shape is `(batch, 1, 6, wavelengths)`.

View order:

```python
["raw", "snv", "raw_sg1", "snv_sg1", "raw_sg2", "snv_sg2"]
```

Savitzky-Golay settings are `window_length=21`, `polyorder=3`.

## Colab Setup

Each `T11*.ipynb` experiment notebook is self-contained. The first code cell:

- clones or pulls `https://github.com/2Kentaro1/wood-moisture-2d-cnn.git`
- mounts Google Drive
- sets `PROJECT_ROOT=/content/wood-moisture-2d-cnn`
- sets `OUTPUT_DIR=/content/drive/MyDrive/wood-moisture-2d-cnn-outputs`
- installs `requirements.txt`

`notebooks/T00_colab_setup.ipynb` is kept as an optional setup-only notebook, but it is no longer required before running experiments.

Models, metrics, OOF files, plots, and `temp/` checkpoints are saved to Google Drive, so they survive Colab runtime disconnects.

Make sure `data/train.csv` and `data/test.csv` exist after cloning. If the CSV files are not committed to GitHub, upload them to `/content/wood-moisture-2d-cnn/data/` in Colab before training.

## Training

```bash
python -m src.training.train_regression --task mc --epochs 80
python -m src.training.train_classification --task species --epochs 80
python -m src.training.train_classification --task woodtype --epochs 80
python -m src.training.train_classification --task wood_structure --epochs 80
python -m src.training.train_regression --task index_norm --epochs 80
python -m src.training.train_regression --task mc_norm --epochs 80
```

Cross validation uses `GroupKFold` with `species number` as the group. Outputs are saved under `OUTPUT_DIR` when it is set, otherwise under local `outputs/`.

## Wood Labels

`woodtype` uses the provided species table as a fixed mapping:

- `hardwood`: ウエンジ, ウォールナット, クリ, チェリー, トチ, ナラ, ホワイトオーク
- `softwood`: イチョウ, スプルース, ヒノキ, ベイスギ, ベイマツ, 米ヒバ

`wood_structure` labels are:

- `tracheid`: イチョウ, スプルース, ヒノキ, ベイスギ, ベイマツ, 米ヒバ
- `ring_porous`: クリ, ナラ, ホワイトオーク
- `diffuse_porous`: ウォールナット, チェリー, トチ
- `ring_porous_like`: ウエンジ

## Resume / Skip Behavior

Training is restart-friendly. You can rerun the same notebook or command from the top.

- If `{OUTPUT_DIR}/metrics/{task}_metrics.json`, `{OUTPUT_DIR}/oof/{task}_oof.csv`, `{OUTPUT_DIR}/embeddings/{task}_embeddings.parquet`, and `{OUTPUT_DIR}/models/{task}_best.pt` already exist, the whole task is skipped.
- If a task is incomplete but a fold has `{OUTPUT_DIR}/models/{task}_fold{fold}.pt` plus temp fold outputs, that fold is skipped and reused.
- During training, every epoch writes `{OUTPUT_DIR}/temp/{task}/fold{fold}_checkpoint.pt`.
- If Colab stops mid-fold, rerunning resumes from that checkpoint.
- Fold validation predictions are cached in `{OUTPUT_DIR}/temp/{task}/fold{fold}_valid_predictions.npz`.

Delete the relevant final output files or `{OUTPUT_DIR}/temp/{task}/` when you intentionally want to rerun an experiment from scratch.

## Interpretability

Implemented methods:

- gradient saliency
- integrated gradients
- wavelength band occlusion: `1000-1300`, `1300-1600`, `1600-1800`, `1800-2000`, `2000-2300`, `2300-2500`
- channel/view occlusion
- view x wavelength heatmaps
- task comparison heatmaps
- difference maps
- species-wise and moisture-band aggregation

The interpretation target is to connect model attention with water absorption, transport, scattering, structure exposure, phase shift, and free/bound water behavior.
