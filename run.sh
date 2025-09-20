#!/bin/bash
# Helper script to run the Heart Disease Analysis program
# This script activates the Python virtual environment and runs the analysis

echo "Activating Python virtual environment..."
source .venv/bin/activate

echo "Running Heart Disease Analysis..."
python HW1_Code.py

echo "Analysis complete! Check kmeans_clusters_age_maxhr_final.png for visualization."
