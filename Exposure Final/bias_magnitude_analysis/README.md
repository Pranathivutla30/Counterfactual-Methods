# Bias Magnitude Analysis

This folder implements the **Bias Magnitude Analysis** from the thesis (Section 5, Table 5 and Figure 1).

## Definitions

- **Bias_M(π)** = M_biased(π) − M_unbiased(π) (Eq. 11)
- **Rating metrics:** M ∈ {RMSE, MAE}
- **Ranking metrics:** M ∈ {NDCG@K, Recall@K}, K ∈ {5, 10}
- **Unbiased sources:** Yahoo/Coat = randomized exposure subset; KuaiRec = near-complete observation matrix
- **Ranking distortion:** Kendall's τ between model rankings (biased vs unbiased)
- **Uncertainty:** 95% CI via user-level bootstrap (or bootstrap over models when per-user data unavailable)

## Inputs

1. **Biased metrics:** Collected from `../biased data/results/metrics_*.txt` and `../biased data/{yahoo,coat shopping,kuairec}/biased_results_*.csv` (run the biased-data notebooks and call `compute_and_save_biased_results_*()` to generate the CSVs).
2. **Unbiased metrics:** Either run the generator script (recommended) or fill by hand.
   - **Recommended:** From `pranathi3` run:  
     `python bias_magnitude_analysis/generate_unbiased_metrics.py`  
     This loads each saved model from the biased-data folders and evaluates it on:  
     **Yahoo** = `data/yahoo_data/random.txt`, **Coat** = `data/.../coat/test.csv`, **KuaiRec** = 80/20 test split from `kuairec_combined.csv`.  
     Requires having run the biased-data notebooks (MF, NCF, LightGCN) and saved the `.keras` models.  
     Output: `data/unbiased_metrics.csv` (columns: Dataset, Architecture, Model, RMSE, MAE, NDCG@5, NDCG@10, Recall@5, Recall@10).
   - **Manual:** Fill `data/unbiased_metrics.csv` with the same columns after evaluating models on the unbiased sets.
3. **Relative ESS (optional):** Fill `data/relative_ess_by_dataset.csv` (Dataset, Architecture, relative_ESS) from the ESS notebooks for Figure 1.

## Outputs

- **Table 5:** Bias magnitude (signed and absolute) by dataset, aggregated over MF, NCF, LightGCN.
- **Figure 1:** |Bias_NDCG@10| vs relative ESS (placeholder if ESS not provided).
- **Kendall's τ** and **95% CI** for bias and rank correlation.

Run `Bias_Magnitude_Analysis.ipynb` to produce these.
