#!/bin/bash
# Setup script for nn-UNet environment variables

# Modify these paths according to your setup
export nnUNet_raw="/home/ajoshi/nnUNet_raw"
export nnUNet_preprocessed="/home/ajoshi/nnUNet_preprocessed"
export nnUNet_results="/home/ajoshi/nnUNet_results"

# Create directories if they don't exist
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

echo "nnUNet environment variables set:"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"
echo ""
echo "To make these permanent, add these exports to your ~/.bashrc or ~/.zshrc"
