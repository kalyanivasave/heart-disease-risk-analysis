# Heart Disease Analysis

This program performs heart disease classification using K-Nearest Neighbors (KNN) and clustering analysis using K-means. It analyzes patient data to predict heart disease and identify patterns in age vs. maximum heart rate.

## Files
- `HW1_Code.py`: Main analysis script
- `Heart_Failure.csv`: Dataset containing patient information
- `requirements.txt`: List of required Python packages
- `run.sh`: Helper script to run the analysis

## Setup Instructions

1. Create a new Python virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Analysis

There are two ways to run the analysis:

1. Using the helper script (recommended):
   ```bash
   chmod +x run.sh  # Make script executable (only needed once)
   ./run.sh
   ```

2. Manually:
   ```bash
   source .venv/bin/activate  # Activate virtualenv
   python HW1_Code.py        # Run analysis
   ```

## Expected Output

The program will:
1. Run KNN classification with different K values (3, 9, 21)
2. Print validation accuracies for each K
3. Show the best K value and its performance
4. Display the confusion matrix for the test set
5. Generate a plot (`kmeans_clusters_age_maxhr_final.png`) showing K-means clustering of Age vs. MaxHR

## Troubleshooting

1. If you see "ModuleNotFoundError":
   - Make sure you've activated the virtualenv
   - Run `pip install -r requirements.txt`

2. If you see "FileNotFoundError":
   - Ensure `Heart_Failure.csv` is in the same directory as `HW1_Code.py`

3. If the plot isn't generated:
   - Check if you have write permissions in the directory
   - Ensure matplotlib is installed correctly
