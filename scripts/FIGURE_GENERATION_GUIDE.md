# Figure Generation Guide for Paper

This guide documents how to generate publication-quality figures for the brain tumor segmentation paper.

## Generated Figures

The paper requires 5 figures:

1. **Train vs Validation Dice Curve** — `figures/train_val_dice.png`
2. **Loss Curve** — `figures/loss_curve.png`
3. **Bar Graph of Experimental Results** — `figures/exp_bar.png` ✓ (already generated)
4. **WT/TC/ET Dice Comparison** — `figures/wt_tc_et.png` ✓ (already generated)
5. **Qualitative Overlays (2 samples)** — `figures/qualitative_1.png`, `figures/qualitative_2.png` ✓ (already generated)

## Quick Start

### After Training (when you have `results/gli_multimodal_mvp_seed42/history.json`)

Generate all figures from training history:

```bash
python scripts/generate_figures.py \
  --results-dir results/gli_multimodal_mvp_seed42 \
  --output-dir figures/
```

This will create:
- `figures/train_val_dice.png`
- `figures/loss_curve.png`
- `figures/exp_bar.png`
- `figures/wt_tc_et.png`

### Generate Qualitative Overlays

If you have a trained checkpoint and want to generate real overlays:

```bash
python scripts/generate_qualitative.py \
  --checkpoint-path checkpoints/gli_multimodal_mvp_seed42/best.pt \
  --results-dir results/gli_multimodal_mvp_seed42 \
  --output-dir figures/ \
  --n-samples 2
```

*Note: Currently generates mock overlays. For real overlays, you need to add data loading and inference code.*

## File Details

### `scripts/generate_figures.py`

**Reads:** `results/<run>/history.json` (created during training)

**Generates:**
- `train_val_dice.png` — Epochs vs Dice for training and validation
- `loss_curve.png` — Epochs vs Loss for training and validation
- `exp_bar.png` — Bar chart comparing mean Dice across all experimental stages
- `wt_tc_et.png` — Grouped bars for WT, TC, ET region comparison

**Usage:**
```bash
python scripts/generate_figures.py \
  --results-dir <path_to_results> \
  --output-dir figures/
```

### `scripts/generate_qualitative.py`

**Generates:** Qualitative segmentation overlay figures with 3-panel layout:
- Column 1: Input MRI slice
- Column 2: Ground truth masks
- Column 3: Prediction overlays

**Usage:**
```bash
python scripts/generate_qualitative.py \
  --checkpoint-path <path_to_best.pt> \
  --output-dir figures/ \
  --n-samples 2
```

## Integration with Training

After running the trainer:

```bash
# Train the model
python src/train_multimodal_gli.py \
  --epochs 50 \
  --batch-size 1 \
  --checkpoint-dir checkpoints/gli_multimodal_mvp_seed42 \
  --results-dir results/gli_multimodal_mvp_seed42 \
  --device cuda:0 \
  --amp

# Generate all paper figures
python scripts/generate_figures.py \
  --results-dir results/gli_multimodal_mvp_seed42 \
  --output-dir figures/

# Generate qualitative figures
python scripts/generate_qualitative.py \
  --checkpoint-path checkpoints/gli_multimodal_mvp_seed42/best.pt \
  --output-dir figures/
```

## What History.json Contains

The trainer saves per-epoch metrics in `history.json`:

```json
{
  "epochs": [0, 1, 2, ...],
  "train_loss": [0.85, 0.74, 0.62, ...],
  "val_loss": [0.92, 0.81, 0.69, ...],
  "train_dice": [0.42, 0.53, 0.61, ...],
  "val_dice": [0.38, 0.50, 0.58, ...]
}
```

## LaTeX Integration

All figures are already referenced in `Paper Material/main.tex`:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\linewidth]{figures/train_val_dice.png}
\includegraphics[width=0.48\linewidth]{figures/loss_curve.png}
\caption{(left) Training and validation Dice over epochs. (right) Training and validation loss over epochs.}
\label{fig:curves}
\end{figure}
```

Make sure the `figures/` folder is in the same directory as your main LaTeX file or adjust the path accordingly.

## Figure Quality Requirements

All figures are generated at:
- **DPI:** 300 (publication quality)
- **Format:** PNG
- **Size:** 8-10 inches width, appropriate height for content

These specifications ensure high-quality output suitable for journal submission.

## Customization

Edit the scripts to:
- Adjust line styles, colors, fonts
- Change marker sizes or plot ranges
- Add error bars or confidence intervals
- Modify figure dimensions

## Troubleshooting

**Issue:** "history.json not found"
- **Cause:** Training hasn't completed yet
- **Solution:** Run trainer first to generate history.json

**Issue:** "checkpoint not found"
- **Cause:** Model checkpoint path is incorrect
- **Solution:** Verify checkpoint path and ensure training has completed

**Issue:** Figure colors don't match paper style**
- **Cause:** Default matplotlib colors
- **Solution:** Edit the `colors` variable in generate_figures.py

## Future Enhancements

- Add real data loading and inference for qualitative figures
- Add error bars / confidence intervals
- Support multi-seed aggregation
- Generate LaTeX table data automatically
