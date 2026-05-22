from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PROJECT_ROOT / "outputs" / "T12E_colab_bundle" / "NIR_CNN_2D_T12E_colab"


FILES_TO_COPY = [
    ("data/train.csv", "data/train.csv"),
    ("data/test.csv", "data/test.csv"),
    ("notebooks/T12E_cnn2d_interpretable_features_stepwise_svr.ipynb", "notebooks/T12E_cnn2d_interpretable_features_stepwise_svr.ipynb"),
    ("outputs/T12_cnn_soft_routing/oof/train_woodtype_oof_probs.csv", "outputs/T12_cnn_soft_routing/oof/train_woodtype_oof_probs.csv"),
    ("outputs/T12_cnn_soft_routing/test/test_woodtype_probs.csv", "outputs/T12_cnn_soft_routing/test/test_woodtype_probs.csv"),
    ("src/__init__.py", "src/__init__.py"),
    ("src/config/__init__.py", "src/config/__init__.py"),
    ("src/config/settings.py", "src/config/settings.py"),
    ("src/data/__init__.py", "src/data/__init__.py"),
    ("src/data/preprocessing.py", "src/data/preprocessing.py"),
    ("src/utils/__init__.py", "src/utils/__init__.py"),
    ("src/utils/plotting.py", "src/utils/plotting.py"),
]


README_TEXT = """# T12E Colab Bundle

このフォルダを Google Drive にアップロードして、Colab から参照してください。

推奨配置:

```text
MyDrive/
  NIR_CNN_2D_T12E_colab/
    data/
    notebooks/
    outputs/T12_cnn_soft_routing/
    src/
```

Colab冒頭で以下を実行:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/NIR_CNN_2D_T12E_colab
```

その後、`notebooks/T12E_cnn2d_interpretable_features_stepwise_svr.ipynb` を開くか、
Colab上にアップロードして実行してください。

成果物は Colab 実行時に以下へ保存されます:

```text
MyDrive/
  NIR_CNN_2D_T12E_results/
    T12E_cnn2d_interpretable_stepwise/
      figures/
      submission.csv
      T12E_*.csv
```

入力bundleと成果物フォルダを分けているので、実験を再実行しても元データを汚しません。
"""


def copy_file(src_rel: str, dst_rel: str) -> None:
    src = PROJECT_ROOT / src_rel
    dst = BUNDLE_ROOT / dst_rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src_rel} -> {dst.relative_to(PROJECT_ROOT)}")


def main() -> None:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    for src_rel, dst_rel in FILES_TO_COPY:
        copy_file(src_rel, dst_rel)

    readme = BUNDLE_ROOT / "README_COLAB_T12E.md"
    readme.write_text(README_TEXT, encoding="utf-8")
    print("wrote", readme.relative_to(PROJECT_ROOT))
    print("bundle root:", BUNDLE_ROOT)


if __name__ == "__main__":
    main()
