# CV wording — Research / ML Intern (DeepHealth)

Add to **Selected Projects** after the project is publicly accessible on GitHub. Do not describe this as a peer-reviewed publication or clinical study.

**Data-Efficient Vision Transformer Adaptation for Chest X-Ray Classification** | PyTorch · Hugging Face · PEFT/LoRA · MedMNIST+

- Benchmarked frozen-backbone linear probing against LoRA (`r=8`) for ViT-B/16 on **400 vs 1,200 labeled pediatric chest X-rays**, with validation-selected checkpoints and a held-out test split.
- Achieved **0.978 test AUROC with LoRA on 1,200 images**, compared with **0.950** for linear probing; reported sensitivity, specificity, and balanced accuracy at a fixed operating threshold.
- Built a reproducible Colab workflow with **four training runs, experiment logs, sanity tests, ROC visualizations, and parameter/runtime comparisons**; documented single-seed and dataset limitations.

**Compact two-line version (for a one-page CV):**

- Evaluated ViT-B/16 transfer learning on PneumoniaMNIST+ with Linear Probing vs LoRA over 400/1,200 labeled images, using PyTorch, PEFT and validation-selected checkpoints.
- LoRA reached **0.978 held-out test AUROC** on 1,200 images (vs **0.950** for Linear Probing); documented sensitivity/specificity trade-offs and computational cost in a reproducible GitHub benchmark.

**For interviews:** At 400 images, LoRA improved AUROC but reduced balanced accuracy (0.767 vs 0.791), due to lower specificity at the fixed 0.5 threshold. No clinical claims or statistical-superiority claims from a single seed.
