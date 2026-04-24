# Results Comparison Summary

## Overview
Two reporting approaches have been generated for your thesis:

1. **LightGCN Results Only** - Reports results from the standard LightGCN model
2. **Average Across All Models** - Reports the average of NCF, MF, and LightGCN results per dataset

---

## Option 1: LightGCN Results Only (Standard Model)

**Rationale**: LightGCN is a more standard/commonly used model in recent research.

### Key Findings:

#### Yahoo Dataset:
- **Best Method**: CRM (RMSE: 0.306285, Improvement: 0.01%)
- **Best Ranking**: Naive ERM (NDCG@10: 0.834610)
- **Overall**: Debiasing methods show minimal improvement over baseline

#### Coat Dataset:
- **Best Method**: DR (RMSE: 0.487298, Improvement: 0.65%)
- **Best Ranking**: Naive ERM (NDCG@10: 0.797862)
- **Overall**: DR shows the best RMSE improvement

#### KuaiRec Dataset:
- **Best Method**: CRM (RMSE: 0.002910, Improvement: 1.44%)
- **Best Ranking**: IPS (NDCG@10: 0.457300)
- **Overall**: CRM shows the best improvement, IPS best for ranking

---

## Option 2: Average Across All Models (NCF, MF, LightGCN)

**Rationale**: Provides a more robust estimate by averaging across multiple model architectures.

**Note**: 
- All datasets: Average of 3 models (NCF, MF, LightGCN)

### Key Findings:

#### Yahoo Dataset:
- **Best Method**: DR (RMSE: 0.307357, Improvement: -0.01%)
- **Overall**: All debiasing methods show slight degradation vs baseline

#### Coat Dataset:
- **Best Method**: IPS (RMSE: 0.436493, Improvement: 0.68%)
- **Overall**: All debiasing methods show improvement, with IPS being best

#### KuaiRec Dataset:
- **Best Method**: CRM (RMSE: 0.002911, Improvement: 1.63%)
- **Overall**: CRM shows significant improvement, IPS also performs well (1.07%)

---

## Comparison Table: RMSE Improvement (%)

| Dataset | Method | LightGCN Only | Average (All Models) |
|---------|--------|---------------|---------------------|
| **Yahoo** | IPS | -0.56% | -0.09% |
| | SNIPS | -0.16% | -0.20% |
| | CRM | 0.01% | -0.28% |
| | DR | -0.21% | -0.01% |
| **Coat** | IPS | 0.22% | **0.68%** |
| | SNIPS | 0.28% | 0.62% |
| | CRM | -0.13% | 0.27% |
| | DR | **0.65%** | 0.46% |
| **KuaiRec** | IPS | 1.11% | **1.07%** |
| | SNIPS | 0.08% | 0.00% |
| | CRM | **1.44%** | **1.63%** |
| | DR | 0.11% | 0.04% |

---

## Recommendations

### Use **LightGCN Only** if:
- You want to report results from a single, standard model
- You prefer consistency with recent research (LightGCN is widely used)
- You want to avoid averaging artifacts
- Your focus is on model-specific insights

### Use **Average Across Models** if:
- You want more robust, generalizable results
- You want to reduce variance from model-specific effects
- You prefer ensemble-like reporting
- Your focus is on debiasing method effectiveness across architectures

---

## Files Generated

1. `lightgcn_results_only.txt` - Text report of LightGCN results
2. `lightgcn_results_only.csv` - CSV file of LightGCN results
3. `average_across_all_models.txt` - Text report of averaged results
4. `average_across_all_models.csv` - CSV file of averaged results

---

## Next Steps

Review both reports and decide which approach aligns better with your thesis goals and your professor's expectations.

