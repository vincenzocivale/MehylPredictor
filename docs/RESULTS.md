# Results

| Run | Train CpG × Val Sample | Val CpG × Train Sample | Val CpG × Val Sample |
|---|---:|---:|---:|
| V0 4 ep | 0.5055 / 0.0251 | 0.4705 / 0.0217 | 0.4695 / 0.0217 |
| V0 25 ep | 0.5505 / 0.0235 | 0.5295 / 0.0204 | 0.5169 / 0.0207 |
| V3 prior fix | 0.5609 / 0.0150 | 0.5276 / 0.0204 | 0.5135 / 0.0207 |
| V2 + locus PCC | 0.5773 / 0.0147 | 0.5647 / 0.0199 | 0.5342 / 0.0204 |
| V1 + variance normalization | **0.5811 / 0.0144** | **0.5708 / 0.0197** | **0.5401 / 0.0201** |
| MethylProphet paper | 0.5455 / 0.0199 | 0.4194 / 0.0266 | 0.3904 / 0.0271 |

Cells are `MAS-PCC / MSE`.

Protocol caveat: the current canonical bundle reproduces an 8260/918 Array
split, while the paper reports 8258/920 after excluding Array/WGBS patient
overlap. See `BENCHMARK_TABLE5.md`.
