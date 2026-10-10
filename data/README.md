# Data preparation

The experiments use the public FORCE 2020 well-log lithology dataset available from https://zenodo.org/records/4351156.

Prepare the following files:

```text
data/
|-- split_manifest.csv
|-- train.csv
|-- val.csv
|-- test.csv
`-- penalty_matrix.npy
```

The CSV delimiter is a semicolon. The required columns are:

```text
WELL
DEPTH_MD
X_LOC
Y_LOC
GROUP
FORMATION
GR
RDEP
RMED
RSHA
RHOB
NPHI
DTC
SP
CALI
PEF
FORCE_2020_LITHOFACIES_LITHOLOGY
```

`Z_LOC` is optional. When it is absent, the preprocessing pipeline uses negative measured depth for the stored vertical coordinate. Graph candidate search uses the mean measured depth of each supernode.

The lithology order is:

```text
0  30000  Sandstone
1  65030  Sandstone/Shale
2  65000  Shale
3  80000  Marl
4  74000  Dolomite
5  70000  Limestone
6  70032  Chalk
7  88000  Halite
8  86000  Anhydrite
9  99000  Tuff
10 90000  Coal
11 93000  Basement
```

`penalty_matrix.npy` must have shape 12 x 12 and use this class order for both axes.

The fixed well assignment is provided in `split_manifest.csv`. It contains 98 training wells, 10 validation wells, and 10 test wells. Each well appears once, and the three subsets are disjoint. The `WELL` column stores the original well name and the `split` column identifies its assigned subset.

The repository sample files provide a small execution check and do not replace the complete dataset.
