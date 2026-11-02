# Experimental findings and validation checklist

**Study:** ViT-B/16 on PneumoniaMNIST+; Linear Probing vs LoRA, 400 vs 1,200 labeled train images; held-out test size 624. **Run date:** 2026-11-02.

## Verified from the uploaded experiment archive

- `results/metrics.csv` contains **four distinct runs** with valid scalar metrics, correct class totals and consistent balanced-accuracy arithmetic.
- Each confusion matrix contains **624 test images**, with 234 negative (NORMAL) and 390 positive (PNEUMONIA) labels. The denominators are the same for all four rows.
- `results/run_summaries.csv` and `results/validation_history.csv` agree on the best validation AUROC and selected epoch for each of the four runs.
- `results/run_config.json` contains exactly **400** and **1,200** distinct in-range training indices, with the 400-index set nested in the 1,200-index set.
- The supplied PNGs depict the expected ROC, AUROC-vs-size, training efficiency and confusion-matrix comparisons; the ROC curves **cannot be numerically reconstructed from the aggregate CSV alone**, because per-image predicted scores were not exported.
- Five original offline sanity tests are provided, and result-integrity tests were added to check exported aggregate consistency. These tests do **not** establish patient-level independence or clinical validity.

## Primary outcome: discrimination (AUROC)

| Size | Linear Probing AUROC | LoRA AUROC | LoRA − LP |
|--:|--:|--:|--:|
| 400 | 0.931460 | 0.973866 | +0.042406 |
| 1,200 | 0.949737 | 0.978117 | +0.028380 |

*Descriptive comparisons based on a single seed; no confidence interval or significance test.*

The AUROC gain from 400 → 1,200 examples is +0.018277 for LP and +0.004252 for LoRA in this run. This suggests that the low-label regime may matter differently for the two adaptations, but the sample sizes, initialization, optimizer schedules and repeated-run uncertainty are not separately controlled.

## Decision threshold: 0.5, fixed before test evaluation

| Run | TN | FP | FN | TP | Sensitivity | Specificity | Balanced accuracy |
|:--|--:|--:|--:|--:|--:|--:|--:|
| LP-400 | 142 | 92 | 10 | 380 | 0.9744 | 0.6068 | 0.7906 |
| LoRA-400 | 128 | 106 | 5 | 385 | 0.9872 | 0.5470 | 0.7671 |
| LP-1200 | 183 | 51 | 20 | 370 | 0.9487 | 0.7821 | 0.8654 |
| LoRA-1200 | 183 | 51 | 9 | 381 | 0.9769 | 0.7821 | 0.8795 |

At 400 images, LoRA produces higher AUROC and sensitivity but **lower** specificity and balanced accuracy than LP. This demonstrates why a rank-based metric alone is not sufficient to characterize a selected operating point. The fixed 0.5 threshold was **not optimized** on the held-out test set. This study does not estimate how the threshold should be chosen for deployment.

## Computational cost

| Run | Trainable params | Share trainable | Optimizer steps | Time, s | Peak GPU allocated, MiB |
|:--|--:|--:|--:|--:|--:|
| LP-400 | 1,538 | 0.00179% | 75 | 50.8 | 493.5 |
| LoRA-400 | 296,450 | 0.34433% | 150 | 115.7 | 1,817.6 |
| LP-1200 | 1,538 | 0.00179% | 450 | 153.1 | 492.9 |
| LoRA-1200 | 296,450 | 0.34433% | 225 | 133.8 | 1,817.6 |

Parameter efficiency is relative to a full ViT: both configurations update under 1% of their corresponding parameters, but LP trains far fewer than LoRA. At 1,200 images, LoRA has a shorter recorded training time only because the validation stopping rules resulted in **225 vs 450 optimizer updates**. Training time and memory measurements are specific to this T4 runtime.

## Caveats / what has not been checked

1. Dataset sample indices are local to the original train split; numerical index ranges cannot establish absence of patient overlap with validation/test.
2. No raw per-image test labels and probabilities are present, so independent recalculation of AUROC/ROC points from predictions is impossible without rerunning.
3. Checkpoint weights are excluded; the source pipeline will download the public pretrained weights and train again when executed.
4. No independent seed repetitions or uncertainty intervals are available.
5. These experimental observations should not be marketed as evidence of clinical effectiveness, robust OOD generalization, or novel algorithmic research.

## Status

- Completed: four training runs; held-out test evaluations; plots and exported CSV/JSON artifacts; source code and offline sanity checks.
- Not completed: independent rerun from a fresh runtime; model checkpoint preservation; external/patient-level validation; BiomedCLIP extension (optional and intentionally excluded).
