# Method-to-code alignment

| Manuscript operation | Implementation |
|---|---|
| Missing-value indicators and within-well median imputation | `src/SESG_Net/preprocessing.py` |
| Resistivity log transformation | `safe_log10_series` in `preprocessing.py` |
| Within-well standardization and derived responses | `preprocess_split` in `preprocessing.py` |
| Centered sequence windows | `SegmentLithologySequenceDataset` in `dataset.py` |
| Multi-scale sequence encoding | `InceptionBlock1D` and `InceptionResidualBlock` in `model.py` |
| Attention pooling | `AttentionPool1D` in `model.py` |
| Stratigraphic mapping and supernodes | supernode functions in `preprocessing.py` |
| Candidate graph construction | `build_candidate_graph` in `preprocessing.py` |
| Dynamic edge scoring and neighbor aggregation | `DynamicEdgeScorer` and `DynamicWeightedGraphSAGE` in `model.py` |
| Sequence-spatial-stratigraphic fusion | `SegmentSpatialTemporalStratModel.forward` in `model.py` |
| AdamW, cosine annealing, gradient clipping | `train_model` in `training.py` |
| Accuracy, macro F1-score, and Penalty | `train_model` in `training.py` and `run_test` in `experiments/testing/evaluate.py` |

The strict validation protocol uses training wells as graph information sources for validation targets. Validation labels are used only for checkpoint selection and metric reporting.
