# Copyright (c) 2018 John Garrett
# SPDX-License-Identifier: MIT
# See plot/SCIENCEPLOTS_LICENSE.txt.

import matplotlib.pyplot as plt

from .styles_discovery import read_styles_in_folders
from pathlib import Path

# register the bundled stylesheets in the matplotlib style library
scienceplots_path = Path(__file__).resolve().parent
styles_path = scienceplots_path / "styles"

# Reads styles in /styles folder and all subfolders
stylesheets = read_styles_in_folders(str(styles_path))

# Update dictionary of styles - plt.style.library
plt.style.core.update_nested_dict(plt.style.library, stylesheets)
# Update `plt.style.available`, copy-paste from:
# https://github.com/matplotlib/matplotlib/blob/a170539a421623bb2967a45a24bb7926e2feb542/lib/matplotlib/style/core.py#L266  # noqa: E501
plt.style.core.available[:] = sorted(plt.style.library.keys())
