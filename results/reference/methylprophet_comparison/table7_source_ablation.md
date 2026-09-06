# Comparison: MethylProphet vs Ours

> **Historical provenance note**: `RNAMethylationPredictor` and the `MethylProphetTrainer`
> that produced the "Ours" numbers below have since been removed from the codebase (see
> `CLAUDE.md`'s "Model compatibility note"). These numbers remain frozen, valid
> paper-comparison provenance; they are not reproducible by current code.

- **MethylProphet**: published results from the paper
- **Ours**: `RNAMethylationPredictor`, chr1, verified from `evaluation/headline.json`
- **↑** higher is better
- **↓** lower is better


## 1. Train CpG × Val Sample

| Train Data | Val Data | Model | MAS-PCC ↑ | MAC-PCC ↑ | MSE ↓ | MAE ↓ |
|---|---|---|---:|---:|---:|---:|
| T(A) | T(A) | MethylProphet | 0.4000 | 0.8669 | 0.0363 | 0.1216 |
| T(A) | T(A) | **Ours** | **0.6438** | **0.9581** | **0.0124** | **0.0649** |
| T(A+W) | T(A) | MethylProphet | 0.4705 | 0.3244 | 0.2981 | 0.9112 |
| T(A+W) | T(A) | **Ours** | **0.6276** | **0.9565** | **0.0130** | **0.0664** |
| T(A+E) | T(A) | MethylProphet | 0.5226 | 0.9232 | 0.0222 | 0.0920 |
| T(A+E) | T(A) | **Ours** | **0.6443** | **0.9585** | **0.0124** | **0.0651** |
| T(A+E+W) | T(A) | MethylProphet | 0.5455 | 0.9320 | 0.0199 | 0.0882 |
| T(A+E+W) | T(A) | **Ours** | **0.6327** | **0.9569** | **0.0128** | **0.0661** |


## 2. Val CpG × Train Sample

| Train Data | Val Data | Model | MAS-PCC ↑ | MAC-PCC ↑ | MSE ↓ | MAE ↓ |
|---|---|---|---:|---:|---:|---:|
| T(A) | T(A) | MethylProphet | 0.2769 | 0.7914 | 0.0555 | 0.1498 |
| T(A) | T(A) | **Ours** | **0.5906** | **0.9301** | **0.0196** | **0.0859** |
| T(A+W) | T(A) | MethylProphet | **0.8674** | 0.8673 | 0.0252 | **0.0365** |
| T(A+W) | T(A) | **Ours** | 0.5882 | **0.9304** | **0.0196** | 0.0859 |
| T(A+E) | T(A) | MethylProphet | 0.3727 | 0.8738 | 0.0350 | 0.1147 |
| T(A+E) | T(A) | **Ours** | **0.6026** | **0.9339** | **0.0185** | **0.0836** |
| T(A+E+W) | T(A) | MethylProphet | 0.4194 | 0.9065 | 0.0266 | 0.1000 |
| T(A+E+W) | T(A) | **Ours** | **0.5984** | **0.9334** | **0.0187** | **0.0841** |


## 3. Val CpG × Val Sample

| Train Data | Val Data | Model | MAS-PCC ↑ | MAC-PCC ↑ | MSE ↓ | MAE ↓ |
|---|---|---|---:|---:|---:|---:|
| T(A) | T(A) | MethylProphet | 0.2597 | 0.7930 | 0.0557 | 0.1504 |
| T(A) | T(A) | **Ours** | **0.5507** | **0.9273** | **0.0204** | **0.0879** |
| T(A+W) | T(A) | MethylProphet | 0.0369 | 0.1006 | 0.1205 | 0.1212 |
| T(A+W) | T(A) | **Ours** | **0.5469** | **0.9282** | **0.0203** | **0.0878** |
| T(A+E) | T(A) | MethylProphet | 0.3451 | 0.8743 | 0.0355 | 0.1157 |
| T(A+E) | T(A) | **Ours** | **0.5656** | **0.9316** | **0.0192** | **0.0856** |
| T(A+E+W) | T(A) | MethylProphet | 0.3904 | 0.9059 | 0.0271 | 0.1011 |
| T(A+E+W) | T(A) | **Ours** | **0.5613** | **0.9307** | **0.0194** | **0.0859** |