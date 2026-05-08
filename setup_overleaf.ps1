# QUICK SETUP: Copy Figures to Overleaf Folder
# Run these commands from PowerShell in the Brain_Tumor_Segmentation folder

# Create figures subfolder in overleaf_submission
mkdir -Path "overleaf_submission\figures" -Force

# Copy the 4 required figures
Copy-Item -Path "figures\exp_bar.png" -Destination "overleaf_submission\figures\exp_bar.png" -Force
Copy-Item -Path "figures\wt_tc_et.png" -Destination "overleaf_submission\figures\wt_tc_et.png" -Force
Copy-Item -Path "figures\qualitative_1.png" -Destination "overleaf_submission\figures\qualitative_1.png" -Force
Copy-Item -Path "figures\qualitative_2.png" -Destination "overleaf_submission\figures\qualitative_2.png" -Force

# Optional: Copy training curves if they exist
Copy-Item -Path "figures\train_val_dice.png" -Destination "overleaf_submission\figures\train_val_dice.png" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "figures\loss_curve.png" -Destination "overleaf_submission\figures\loss_curve.png" -Force -ErrorAction SilentlyContinue

# Verify
Write-Host "Setup complete! Files copied:"
Get-ChildItem "overleaf_submission\figures" | Select-Object Name, Length

# Output folder ready for Overleaf/LaTeX
Write-Host "`nOverleaf folder ready at: overleaf_submission/"
Write-Host "Next step: Upload to Overleaf or compile locally with pdflatex"
