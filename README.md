# Data-Efficient Adaptation of Vision Transformers for Chest X-Ray Classification

**Reproducible applied ML research portfolio** comparing linear probing and Low-Rank Adaptation (LoRA) of an ImageNet-21k pretrained Vision Transformer on pediatric chest X-rays.

**Experiment date:** 2 November 2026 · **Seed:** 42 · **Dataset:** PneumoniaMNIST+ (224×224) · **Evaluation:** official held-out test split (624 images).

> **Research-only demonstration.** These models have not been clinically validated and must not be used for medical diagnosis.

## Research question

How does the amount of labeled medical imaging data affect the predictive performance and computational cost of linear probing versus LoRA adaptation of a pretrained ViT?

## Test results

| Method | Labeled training images | Test AUROC ↑ | Balanced accuracy ↑ | Sensitivity | Specificity | Trainable params | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear probing | 400 | 0.9315 | **0.7906** | 0.9744 | 0.6068 | 1,538 | 50.8 s |
| LoRA | 400 | **0.9739** | 0.7671 | 0.9872 | 0.5470 | 296,450 | 115.7 s |
| Linear probing | 1,200 | 0.9497 | 0.8654 | 0.9487 | 0.7821 | 1,538 | 153.1 s |
| LoRA | 1,200 | **0.9781** | **0.8795** | 0.9769 | 0.7821 | 296,450 | 133.8 s |

Source: real model runs exported from Google Colab, summarized in [`results/metrics.csv`](results/metrics.csv). All four checkpoints were selected based on validation AUROC before test evaluation.

**Key findings**

- In this single-seed study, LoRA improved test AUROC over linear probing by **0.0424** with 400 images and **0.0284** with 1,200 images.
- LoRA-400 (0.9739) outperformed linear-probing-1200 (0.9497) on test AUROC, but the comparison does **not** control optimizer updates, trainable parameters, or compute equally.
- **Higher AUROC is not the same as a better fixed-threshold classification result:** at the predefined threshold 0.5, LoRA-400 had lower balanced accuracy (0.7671 vs 0.7906), caused by poorer specificity.
- LoRA-1200 recorded shorter training time than LP-1200, but used **three vs six epochs**, so this does not demonstrate intrinsically faster training.
- We report observations, not statistical significance: only one seed, one pediatric dataset, two annotation budgets, and one pretrained checkpoint were studied.

![Summary of four study findings](results/figures_overview.svg)

## Methods

- **Data:** 4,708 training, 524 validation, and 624 test images. Nested stratified random subsets (400 inside 1,200), same examples for both methods.
- **Model:** `google/vit-base-patch16-224-in21k`, ViT-B/16. Grayscale X-rays converted to RGB and processed using the pretrained checkpoint's image processor.
- **Linear probing:** encoder frozen and in evaluation mode; external 768-to-2 linear classification head trainable; AdamW learning rate 0.001.
- **LoRA:** frozen backbone with trainable rank-8 adapters on `query` and `value` (alpha 16, dropout 0.1), plus the same external trainable classification head; AdamW learning rate 0.0002.
- **Training:** batch size 16, weighted cross-entropy, FP32, seed 42; pre-registered 3-to-6-epoch validation stopping rule; checkpoint chosen by validation AUROC.
- **Test:** one final held-out evaluation after all four training runs, with AUROC primary and balanced accuracy, sensitivity and specificity calculated at a fixed threshold 0.5.

See [experiment protocol](docs/experimental_protocol.md) and [scientific findings](docs/findings.md).

## Repository contents

- `project.py` — training, validation, final evaluation, figures
- `utils.py` — data sampling, evaluation metrics, early stopping
- `tests/` — pipeline and aggregate-result sanity tests
- `docs/` — methodology, critical analysis and CV-ready description
- `results/` — recorded aggregate metrics, validation history, runtime summaries and visual results

**No notebook is included in this public repository.** The Python scripts are the reproducible implementation.

## How to reproduce

Use a fresh GPU-enabled Python environment with internet access for public weights/data. The recorded experiments ran on Google Colab Tesla T4 (Python 3.13, PyTorch 2.11.0+cu130, Transformers 4.57.6, PEFT 0.21.2, MedMNIST 3.0.2, scikit-learn 1.6.1).

```bash
python -m pip install -r requirements.txt
python -m pip uninstall -y torchao
python -m pytest -q tests
```

Then run the research pipeline in Python:

```python
import project

data = project.prepare_data()
project.run_experiments(data, project.EXPERIMENTS)
final_metrics = project.finalize_results(data)
print(final_metrics)
```

This downloads the public dataset/checkpoint and retrains the four configurations; results may differ across hardware and library versions. Do not use published test outcomes to retune hyperparameters. The test set must not be reused for further model selection.

## Limitations and reproducibility boundaries

The exported results include aggregate performance metrics and training summaries, but **not per-image predictions or trained checkpoint weights**. Thus the exact ROC points and model weights cannot be independently recovered from the aggregate tables alone; rerunning the experiments is required. Patient-level independence across splits was not independently verified. No clinical, external-site, or out-of-distribution validation was carried out.

## References

- [MedMNIST](https://medmnist.com/)
- [ViT-B/16 on Hugging Face](https://huggingface.co/google/vit-base-patch16-224-in21k)
- [PEFT / LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
