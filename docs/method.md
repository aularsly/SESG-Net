# SESG-Net method

## Input features

The model uses GR, RDEP, RMED, RSHA, RHOB, NPHI, DTC, SP, CALI, and PEF. A binary missing-value indicator is created for each curve before imputation. Missing values are filled with the median of the corresponding curve in each well, with the training-set median used when an entire well curve is unavailable.

RDEP, RMED, and RSHA are transformed with base-10 logarithms. The continuous curves are standardized within each well. The derived responses are calculated as `UMA = PEF x RHOB` and `VSH = (GR - GRmin) / (GRmax - GRmin)`.

## Sequence encoding

A centered window of 129 samples is constructed for each target depth. The sequence encoder uses channel projection, parallel one-dimensional convolutions with kernel sizes 9, 19, and 39, residual feature extraction, and attention pooling. The ten missing-value indicators are processed by an auxiliary sequence branch and fused with the continuous sequence representation.

## Spatial graph

Consecutive samples with similar responses in the same well and stratigraphic unit are grouped into supernodes. Each supernode contains response means, response standard deviations, mean depth, spatial coordinates, stratigraphic identifiers, relative depth, and interval thickness.

Cross-well candidates are found inside a three-dimensional search radius. Formation is used first, followed by group-level candidates when required. Candidate relations contain horizontal distance, vertical distance, relative-depth difference, response-mean difference, and response-standard-deviation difference. The learned edge scorer selects 10 neighbors from the candidate pool and normalizes their scores into aggregation weights.

GraphSAGE combines the transformed target feature with the weighted transformed neighbor features. The resulting supernode representation is assigned to every original sample belonging to that supernode.

## Fusion and training

The sequence, missing-state, and spatial representations are concatenated and passed through the classification head. The network is trained with cross-entropy loss and AdamW. Accuracy is used for checkpoint selection, with Penalty used as the tie breaker.

## Evaluation

The reported metrics are Accuracy, Precision, Recall, F1-score, and Penalty. The class mapping and penalty-matrix order are defined in `src/SESG_Net/config.py` and `data/README.md`.
