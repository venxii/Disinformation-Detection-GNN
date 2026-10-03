
### 4class

| held-out event | n | train-majority class | majority acc | DGTR acc @10% | DGTR acc @100% | GAT acc @100% | DGTR beats majority? | DGTR mF1 @100% | majority mF1 |
|---|---|---|---|---|---|---|---|---|---|
| charliehebdo | 2079 | non-rumour | 0.780 | 0.658 | 0.678 | 0.671 | **no** | 0.342 | 0.219 |
| ebola-essien | 14 | non-rumour | 0.000 | 0.143 | 0.214 | 0.143 | yes | 0.088 | 0.000 |
| ferguson | 1143 | non-rumour | 0.752 | 0.687 | 0.696 | 0.716 | **no** | 0.254 | 0.215 |
| germanwings-crash | 468 | non-rumour | 0.491 | 0.342 | 0.340 | 0.316 | **no** | 0.276 | 0.165 |
| gurlitt | 138 | non-rumour | 0.558 | 0.174 | 0.174 | 0.413 | **no** | 0.109 | 0.179 |
| ottawashooting | 890 | non-rumour | 0.472 | 0.615 | 0.621 | 0.522 | yes | 0.397 | 0.160 |
| prince-toronto | 233 | non-rumour | 0.017 | 0.082 | 0.052 | 0.052 | yes | 0.045 | 0.008 |
| putinmissing | 238 | non-rumour | 0.471 | 0.298 | 0.336 | 0.349 | **no** | 0.188 | 0.160 |
| sydneysiege | 1221 | non-rumour | 0.572 | 0.552 | 0.538 | 0.509 | **no** | 0.365 | 0.182 |

Pooled accuracy: DGTR 0.575, GAT 0.561, train-majority 0.626. Pooled macro-F1: DGTR 0.358, GAT 0.345, majority 0.193. DGTR beats majority accuracy in 3/9 folds.

### veracity

| held-out event | n | train-majority class | majority acc | DGTR acc @10% | DGTR acc @100% | GAT acc @100% | DGTR beats majority? | DGTR mF1 @100% | majority mF1 |
|---|---|---|---|---|---|---|---|---|---|
| charliehebdo | 458 | true | 0.421 | 0.402 | 0.393 | 0.384 | **no** | 0.302 | 0.198 |
| ebola-essien | 14 | true | 0.000 | 0.214 | 0.214 | 0.143 | yes | 0.118 | 0.000 |
| ferguson | 284 | true | 0.035 | 0.162 | 0.169 | 0.211 | yes | 0.113 | 0.023 |
| germanwings-crash | 238 | true | 0.395 | 0.416 | 0.399 | 0.324 | yes | 0.315 | 0.189 |
| gurlitt | 61 | true | 0.967 | 0.295 | 0.295 | 0.148 | **no** | 0.174 | 0.328 |
| ottawashooting | 470 | true | 0.700 | 0.364 | 0.298 | 0.370 | **no** | 0.218 | 0.275 |
| prince-toronto | 229 | true | 0.000 | 0.087 | 0.083 | 0.127 | yes | 0.059 | 0.000 |
| putinmissing | 126 | true | 0.000 | 0.079 | 0.111 | 0.310 | yes | 0.092 | 0.000 |
| sydneysiege | 522 | true | 0.732 | 0.352 | 0.320 | 0.303 | **no** | 0.295 | 0.282 |

Pooled accuracy: DGTR 0.285, GAT 0.301, train-majority 0.444. Pooled macro-F1: DGTR 0.266, GAT 0.296, majority 0.205. DGTR beats majority accuracy in 5/9 folds.
