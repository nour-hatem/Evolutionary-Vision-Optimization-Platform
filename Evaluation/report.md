# Model Evaluation Report — Baseline vs GA-Optimized CNN
*Cloud Computing Project*

---

## Results Summary

| Metric | Baseline | GA-Optimized | Delta |
|--------|----------|-------------|-------|
| Accuracy | 0.8587 | 0.8993 | **+0.0406** |
| F1 Macro | 0.8579 | 0.8987 | **+0.0408** |
| F1 Weighted | 0.8579 | 0.8987 | +0.0408 |
| Params | 3,377,866 | 12,249,514 | +8,871,648 |
| Inference (ms/batch) | 56.73 | 48.84 | -7.89 |
| File Size (MB) | 13.53 | 49.04 | +35.51 |

---

## Which Model is Better?

**Winner: GA-Optimized**

The GA-Optimized model achieves **+0.0406 higher accuracy** and **+0.0408 higher Macro F1**.
The F1 improvement being consistent with accuracy means the optimized model is more balanced
across all 10 classes, not just better on easy ones.

---

## Why Did the GA Improve Performance?

The GA ran for 11 generations, evolving hyperparameter sets and selecting the fittest each round.

| Hyperparameter | Baseline | GA-Optimized | Why it helps |
|----------------|----------|-------------|--------------|
| `num_filters` | 64 | 96 | More filters = richer feature maps |
| `num_layers` | 3 | 4 | Deeper network captures more abstract patterns |
| `dropout` | 0.5 | 0.3 | Larger model needs less aggressive regularization |
| `lr` | 0.0005 | 0.0003 | Lower LR = smoother convergence |
| `batch_size` | 64 | 32 | Smaller batches = better gradient estimates |

GA fitness improved from `0.7002` (gen 0) to
`0.7338` (gen 10).

---

## Observations

- **Best improved class:** `bird` (+0.0890)
- **Most regressed class:** `airplane` (0.0010)
- The optimized model is larger and slightly slower at inference but the accuracy gain justifies it.
- GA convergence shows fast early improvement (gen 0–4) then stabilization — a healthy pattern.
- Visually similar classes (cat/dog, automobile/truck) have lowest per-class accuracy in both models.

---
*Dataset: CIFAR-10 | Test images: 10,000 | Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*
