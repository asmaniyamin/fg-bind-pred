# Per-Functional-Group Binding Contribution Prediction

Code for estimating functional-group contributions to protein–ligand binding, using molecular graphs, pocket features, and counterfactual MM/GBSA labels.

## Source layout

| Directory | Purpose |
| --- | --- |
| `models/model1/` | Per-FG contact energy and preorganization model |
| `models/model2/` | Counterfactual importance and hotspot model |
| `dataloader/` | MM/GBSA sample assembly, tensorization, and PyTorch adapters |
| `mmgbsa/` | Baseline MM/GBSA execution and structure repair |
| `scripts/` | FG masks, masked calculations, batch execution, training sweeps, and reports |
| `protein/` | Protein preparation and ligand/pocket extraction |
| `small-molecule-mod-toolkit/` | Integrated chemistry, graph, and functional-group utilities |
| `plot/` | Plot style helpers and vendored plotting assets |

## Environment and entry points

Use Python 3.10 or later. MM/GBSA preprocessing needs the original scientific environment, including AmberTools; see `mmgbsa/README.md`.

Run model commands from the repository root, using an explicit prepared tensor cache and output directory:

```sh
python -m models.model1.train --help
python -m models.model2.train --help
python -m models.model1.train --data-root /path/to/tensor-cache --output-root outputs/model1
python -m models.model2.train --data-root /path/to/tensor-cache --output-root outputs/model2
```

## Citation

If you use this code or derivatives in research, please cite *Per-Functional-Group Binding Contribution Prediction* by Asmani Yamin. See [CITATION.cff](CITATION.cff) for citation metadata.

## License

Copyright 2026 Asmani Yamin.

The project source code is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).

You may use, modify, and distribute the code, including for commercial purposes, subject to the license terms. When distributing the code or derivative works, include the license, preserve applicable copyright and attribution notices (including NOTICE), and identify modified files as required by section 4.

Vendored SciencePlots modules and styles in `plot/scienceplots/` retain their MIT license. See [the SciencePlots license](plot/SCIENCEPLOTS_LICENSE.txt).
