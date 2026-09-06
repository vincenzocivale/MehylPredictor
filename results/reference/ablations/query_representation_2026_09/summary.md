# query_representation_2026_09

Controlled architecture-selection experiment. **Do not report these as official test numbers.**
All arms use the same final locus-attention recipe, training seed, data and fixed inner split; only the query representation changes.

| arm | query | best epoch | double-OOD MAS-PCC | double-OOD MSE | Δ MAS vs Q0 | unseen-CpG/seen-sample MAS |
|---|---|---:|---:|---:|---:|---:|
| `q0_ntv3` | `ntv3` | 13 | 0.5352 | 0.02839 | +0.0000 | 0.5648 |
| `q2_hybrid_detached` | `hybrid_detached` | 12 | 0.5314 | 0.02779 | -0.0038 | 0.5600 |
| `q3_hybrid_joint` | `hybrid_joint` | 7 | 0.5312 | 0.02721 | -0.0040 | 0.5577 |
| `q1_mean_only` | `mean_only` | 12 | 0.5161 | 0.02839 | -0.0191 | 0.5436 |

**Selected on inner double-OOD MAS-PCC:** `q0_ntv3` (`ntv3`).

Next: one final-protocol confirmation of the selected architecture; do not choose again on the official split.
