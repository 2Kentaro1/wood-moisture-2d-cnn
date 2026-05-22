from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_matplotlib_japanese() -> None:
    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "IPAGothic",
        "Yu Gothic",
        "YuGothic",
        "Meiryo",
        "MS Gothic",
        "Hiragino Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
