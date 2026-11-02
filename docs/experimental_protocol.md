# Preregistered benchmark protocol — V3.0

Question: How does labeled training set size impact performance and computational efficiency of linear probing vs LoRA on a pretrained ViT?

Four configurations: LP-400, LoRA-400, LP-1200, LoRA-1200. Same image and split preprocessing, seed 42, same external 2-class classifier architecture, method-specific fixed learning rate across both data budgets.

After epoch 3, extend training if the epoch-3 validation AUROC is the best so far **and** at least 0.003 above epoch 2. During the extension, stop after two successive epochs with <0.002 improvement on the best validation AUROC, or after epoch 6. The best validation AUROC checkpoint is selected. These rules are fixed before using the test split.

Primary metric: ROC-AUC. Secondary metrics: balanced accuracy, sensitivity, specificity at a fixed threshold of 0.5. Compute class weights from each training subset independently. Log train loss and val AUROC each epoch. Report number of optimizer steps, trainable parameters, elapsed training time, and peak allocated GPU memory.

The final test split is evaluated only when all four checkpoints are present. No tuning based on test outcomes. Do not treat one-seed results as evidence of statistical superiority. Patient-level independence cannot be independently verified from the public class labels and numeric indices alone. The work is not clinically validated.

## Post-execution note (2026-11-02)

The four training runs and one final test evaluation were completed according to the recorded protocol. See `results/metrics.csv` and `docs/findings.md`. The test scores were **not used** to revise hyperparameters. The optional vision-language experiment was **not** executed and is not part of this release.
