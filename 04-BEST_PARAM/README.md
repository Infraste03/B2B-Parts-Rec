# 04 - Best Hyperparameter Configurations & Final Results

This directory contains the final, distilled results of the comprehensive Hyperparameter Optimization (HPO) process conducted for each benchmarked model. It serves as the primary summary of the experimental findings.

## Final Benchmark Performance Summary

The table below summarizes the best performance achieved by each model on the test set, using the optimal hyperparameter configuration found during the HPO phase.

| Model      | Recall@20 (LOO)     | NDCG@20 (LOO)       | Recall@20 (Temporal) | NDCG@20 (Temporal)  | Type                  |
|------------|---------------------|---------------------|----------------------|---------------------|-----------------------|
| **Sequential Baselines** ||||||
| FPMC       | **0.1607 ± 0.0131** | **0.0779 ± 0.0055** | **0.2989 ± 0.0037**  | **0.2015 ± 0.0021** | Markov Chain          |
| NextItNet  | **0.1900 ± 0.0165** | **0.0991 ± 0.0062** | **0.4744 ± 0.0036**  | **0.3318 ± 0.0016** | CNN-based             |
| GRU4Rec    | **0.2776 ± 0.0056** | **0.1620 ± 0.0045** | **0.5351 ± 0.0036**  | **0.3916 ± 0.0025** | RNN-based             |
| RepeatNet  | **0.2941 ± 0.0193** | **0.1532 ± 0.0121** | **0.4965 ± 0.0021**  | **0.3265 ± 0.0010** | Repeat-Aware          |
| NARM       | **0.2986 ± 0.0082** | **0.1686 ± 0.0071** | **0.5145 ± 0.0016**  | **0.3675 ± 0.0023** | RNN + Attention       |
| BERT4Rec   | **0.3078 ± 0.0206** | **0.1720 ± 0.0124** | **0.5659 ± 0.0008**  | **0.4010 ± 0.0002** | Transformer           |
| **Knowledge/Context-Aware Models** ||||||
| FDSA       | **0.3534 ± 0.0218** | **0.1857 ± 0.0072** | **0.5683 ± 0.0005**  | **0.3868 ± 0.0011** | Context-Aware Transformer |
| **State-of-the-Art Sequential** ||||||
| **SASRec** | **0.3699 ± 0.0103** | **0.1950 ± 0.0085** | **0.5923 ± 0.0010**  | **0.4014 ± 0.0007** | **Transformer**       |



### Non-Personalized Baselines: Global vs. Temporal Popularity

We  include a time-based popularity-based baseline to the experiments

| Window Size         | Recall@20 (LOO) | NDCG@20 (LOO) | Recall@20 (Temporal) | NDCG@20 (Temporal) |
|---------------------|-----------------|---------------|-----------------|---------------|
| **Global (All Time)**| **0.0457**      | **0.0175**   |0.0236 | 0.1719|
| 14 Days             | 0.0365          | 0.0168        |0.0645 | 0.0277|
| 30 Days             | 0.0411          | 0.0155        |0.0755 | 0.0345|
| 60 Days             | 0.0411          | 0.0134        |0.0812 | 0.0369|
| 90 Days             | 0.0274          | 0.0082        |0.0869 | 0.0382|
| 180 Days            | 0.0274          | 0.0083        |0.0965 | 0.0418|
| 365 Days            | 0.0274          | 0.0089        |0.0948 | 0.04225|



---

## Detailed Configurations

This folder also contains one `.json` file for each model (e.g., `sasrec.json`, `bert4rec.json`). Each file documents the single best hyperparameter configuration that produced the results shown in the table above. The structure is as follows:

-   **`model`**: The name of the recommendation model.
-   **`best_params`**: A dictionary containing the specific set of hyperparameters.
-   **`performance`**: A dictionary containing the final performance metrics (`Recall@20` and `NDCG@20`).

This structure ensures the full reproducibility of the reported results.

## Methodology

The "best" configuration for each model was determined by selecting the HPO run that yielded the highest **`Recall@20`** on the validation set, as this was the primary metric for optimization (`valid_metric`) during the search.

The complete scripts for the HPO process that generated these results are available in the `03-BENCHMARKS/` directory.