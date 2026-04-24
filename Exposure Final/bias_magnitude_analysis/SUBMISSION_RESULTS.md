# Bias Magnitude Analysis — Results for Submission

**Thesis:** Bias magnitude (Eq. 11), ranking distortion (Kendall's τ, Eq. 12), 95% CI, Table 5, Figure 1.

---

## 1. Table 5 — Bias magnitude by dataset (mean over MF, NCF, LightGCN)

| Dataset  | Bias_RMSE | \|Bias_RMSE\| | Bias_MAE | \|Bias_MAE\| |
|----------|-----------|----------------|----------|--------------|
| COAT     | -1.9121   | 1.9121         | -1.661   | 1.661        |
| KUAIREC  | -1.9600   | 1.9600         | -0.9426  | 0.9426       |
| YAHOO    | -0.0676   | 0.0676         | -0.0549  | 0.0549       |

*Rating metrics (RMSE, MAE) are reported. Ranking metrics (NDCG@K, Recall@K) are available in the full per-model results; Table 5 here shows the dataset-level means for rating bias.*

---

## 2. Kendall's τ (ranking correlation: biased vs unbiased)

- **RMSE:** τ = -0.6667, p = 0.3333  
- **MAE:** τ = -0.6667, p = 0.3333  
- **NDCG@10 / Recall@10:** Not computed in this run (biased evaluation had rating metrics only for the merged pairs).

*τ = 1 would indicate identical rankings; negative τ indicates ranking distortion under biased evaluation.*

---

## 3. Bootstrap 95% CI for mean bias (over models)

- **Bias_RMSE:** point = -1.0018, 95% CI = [-1.9360, -0.0676]  
- **Bias_MAE:** point = -0.6784, 95% CI = [-1.3018, -0.0549]  

*Model-level bootstrap (n = 2000); user-level bootstrap would require per-user predictions.*

---

## 4. Interpretation (thesis-aligned)

- **YAHOO:** Small bias magnitude (|\|Bias\| ≈ 0.05–0.07 for RMSE/MAE); standard offline evaluation is relatively reliable.  
- **COAT / KUAIREC:** Larger bias magnitude; exposure bias substantially distorts evaluation — counterfactual correction is important for reliable model selection.  
- **Kendall's τ < 1:** Indicates that biased evaluation mis-ranks models compared with unbiased evaluation.

---

## 5. Figure 1 (Bias vs relative ESS)

*Figure 1 in the thesis relates |Bias_NDCG@10| to relative ESS (ESS/N).*  
To produce it: fill `data/relative_ess_by_dataset.csv` with columns `Dataset`, `Architecture`, `relative_ESS` from your ESS notebooks, then re-run **Bias_Magnitude_Analysis.ipynb** (Section 7). The notebook will save `figure1_bias_vs_ess.png`. If you do not have ESS values yet, you can submit the above results and note that Figure 1 will be added once relative ESS is computed.

---

## Files to attach (optional)

- **Table 5 (CSV):** `bias_magnitude_analysis/data/table5_bias_magnitude.csv`  
- **Full bias table (per model):** from running **Bias_Magnitude_Analysis.ipynb** (Section 3 output)  
- **Figure 1:** `figure1_bias_vs_ess.png` (after filling relative ESS and re-running Section 7)
