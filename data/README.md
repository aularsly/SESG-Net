# Data

The experiments use the public [FORCE 2020 well-log lithology dataset](https://zenodo.org/records/4351156).

Place the prepared files in this directory:

```text
train.csv
val.csv
test.csv
penalty_matrix.npy
```

The CSV files use a semicolon (`;`) delimiter. They contain the well identifier, depth, coordinates, stratigraphic fields (`GROUP` and `FORMATION`), well-log curves, and the `FORCE_2020_LITHOFACIES_LITHOLOGY` label. The three CSV files provide the training, validation, and held-out test data, respectively.

`penalty_matrix.npy` is the 12 x 12 geological penalty matrix used for evaluation. Dataset files are kept local and are not committed to the repository.
