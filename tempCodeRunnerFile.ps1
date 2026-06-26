# QUICK SETUP: Copy Figures to Overleaf Folder
# Run these commands from PowerShell in the Brain_Tumor_Segmentation folder

# Ensure we're in the correct directory
Set-Location "D:\Brain_Tumor_Segmentation"

# Create figures subfolder in overleaf_submission
mkdir -Path "overleaf_submission\figures" -Force | Out-Null

# Copy the quantitative analysis figures (charts)
Copy-Item -Path "figures\exp_bar.png" -Destination "overleaf_submission\figures\exp_bar.png" -Force
Copy-Item -Path "figures\wt_tc_et.png" -Destination "overleaf_submission\figures\wt_tc_et.png" -Force

# Copy ACTUAL MODEL OUTPUT QUALITATIVE RESULTS (real MRI with skull and segmentations)
# From the best-performing run (gli_4mod_region_new_tta)
Copy-Item -Path "results\gli_4mod_region_new_tta\visuals\BraTS-GLI-02210-100_slice83.png" -Destination "overleaf_submission\figures\qualitative_1.png" -Force
Copy-Item -Path "results\gli_4mod_region_new_tta\visuals\BraTS-GLI-02587-104_slice68.png" -Destination "overleaf_submission\figures\qualitative_2.png" -Force

# Optional: Copy training curves if they exist
Copy-Item -Path "figures\train_val_dice.png" -Destination "overleaf_submission\figures\train_val_dice.png" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "figures\loss_curve.png" -Destination "overleaf_submission\figures\loss_curve.png" -Force -ErrorAction SilentlyContinue

# Verify
Write-Host "Setup complete! Files copied:"
Get-ChildItem "overleaf_submission\figures" | Select-Object Name, Length
Write-Host "`nNote: Qualitative figures are now REAL MODEL OUTPUTS from gli_4mod_region_new_tta"
Write-Host "showing actual MRI scans with brain and tumor segmentations`n"

# Output folder ready for Overleaf/LaTeX
Write-Host "Overleaf folder ready at: overleaf_submission/"
Write-Host "Next step: Upload to Overleaf or compile locally with pdflatex"
