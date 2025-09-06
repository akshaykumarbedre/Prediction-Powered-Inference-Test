# Prediction-Powered Inference Test

A demonstration of Prediction-Powered Inference (PPI) methodology using the California housing dataset, comparing statistical inference approaches and showcasing the effectiveness of PPI in reducing confidence interval widths.

## Overview

This repository contains a Jupyter notebook that implements and compares three different statistical inference methods:

1. **Prediction-Powered Inference (PPI)** - A modern approach that leverages both labeled and unlabeled data with machine learning predictions
2. **Classical Statistical Inference** - Traditional confidence interval estimation using only labeled data
3. **Imputation-based Inference** - Using machine learning predictions as a substitute for true labels

## What is Prediction-Powered Inference?

Prediction-Powered Inference is a statistical framework that combines:
- A small amount of labeled data (ground truth)
- A large amount of unlabeled data with machine learning predictions
- Statistical theory to provide rigorous confidence intervals

The key insight is that while machine learning predictions may be biased, they often contain valuable signal that can be used to improve statistical inference when combined with a small amount of labeled data for bias correction.

## Repository Contents

- `ppi_test_on_fetch_california_housing.ipynb` - Main analysis notebook using the full PPI methodology
- `demo.py` - Standalone demonstration script showing simplified PPI concepts
- `utils.py` - Utility functions for plotting and analysis
- `output.png` - Generated visualization comparing the three methods
- `requirements.txt` - Python package dependencies
- `SETUP.md` - Detailed setup instructions
- `LICENSE` - MIT license
- `README.md` - This documentation file

## Dataset

The analysis uses the **California Housing Dataset** from scikit-learn, which contains:
- 20,640 samples of California housing prices
- 8 features including median income, house age, average rooms, etc.
- Target variable: median house value for California districts

## Methodology

### Data Split
- **Training Set (30%)**: Used to train a Random Forest regression model
- **Inference Set (70%)**: Split into labeled and unlabeled portions for testing inference methods

### Model Training
- **Algorithm**: Random Forest Regressor (100 estimators)
- **Purpose**: Generate predictions on the inference set to simulate real-world ML deployment scenario

### Inference Comparison
The notebook tests each method across different sample sizes (50 to 500 labeled examples) with 100 trials each:

1. **PPI Method**: Uses labeled examples + unlabeled examples with predictions
2. **Classical Method**: Uses only labeled examples (traditional approach)  
3. **Imputation Method**: Uses only predictions (no labeled data for correction)

## Key Results

The analysis demonstrates that:

- **PPI consistently produces narrower confidence intervals** than classical methods
- **PPI intervals maintain proper coverage** while being more precise
- **Classical methods require more labeled data** to achieve similar precision
- **Pure imputation methods** can be highly biased without labeled data correction

The visualization shows confidence interval widths across different sample sizes, clearly illustrating PPI's efficiency advantage.

## Installation & Setup

### Prerequisites
```bash
# Install required Python packages
pip install numpy pandas scikit-learn tqdm jupyter matplotlib
```

### PPI Library
The main notebook requires the `ppi_py` library. To install:
```bash
pip install ppi_py
```

If `ppi_py` is not available via pip, you may need to install from source:
```bash
git clone https://github.com/aangelopoulos/ppi_py
cd ppi_py
pip install -e .
```

**Note**: If you cannot install `ppi_py`, you can still run the `demo.py` script which demonstrates the core PPI concepts using a simplified implementation.

### Missing Utils Module
The repository now includes a complete `utils.py` file with all necessary plotting and analysis functions.

## Usage

### Quick Demo (Recommended for First-Time Users)

For a quick demonstration without requiring the full `ppi_py` library:

```bash
python demo.py
```

This script:
- Generates synthetic data to avoid network requirements
- Demonstrates simplified PPI concepts 
- Shows ~50% reduction in confidence interval widths
- Creates comparison visualizations
- Runs in under 1 minute

### Full Analysis

1. **Clone the repository**:
   ```bash
   git clone https://github.com/akshaykumarbedre/Prediction-Powered-Inference-Test
   cd Prediction-Powered-Inference-Test
   ```

2. **Install dependencies** (see Installation section above)

3. **Run the demo script**:
   ```bash
   python demo.py
   ```

4. **Run the full notebook** (if ppi_py is available):
   ```bash
   jupyter notebook ppi_test_on_fetch_california_housing.ipynb
   ```

5. **Execute all cells** to reproduce the analysis and generate visualizations

## Expected Output

### Demo Script
The `demo.py` script will:
1. Generate synthetic regression data
2. Train a Random Forest model  
3. Compare PPI vs Classical confidence intervals across sample sizes
4. Print summary statistics showing ~50% width reduction with PPI
5. Generate `ppi_demo_results.png` with comparison plots

### Full Notebook
The notebook will:
1. Load and split the California housing dataset
2. Train a Random Forest model
3. Generate predictions for the inference set
4. Run 100 trials each for sample sizes from 50 to 500
5. Compare confidence intervals across the three methods
6. Generate a visualization showing interval widths vs. sample size

## Understanding the Results

- **Lower interval width = Better precision**
- **PPI intervals should be consistently narrower** than classical intervals
- **The gap between PPI and classical methods** typically decreases as sample size increases
- **Imputation-only results** may show very narrow intervals but could be biased

## Applications

This methodology is valuable for:
- **Scientific research** where labeled data is expensive but predictions are available
- **A/B testing** with machine learning predictions
- **Survey research** combining small representative samples with large prediction datasets
- **Medical studies** leveraging predictive models with limited ground truth data

## Technical Details

### Confidence Level
- All intervals are constructed at 95% confidence level (α = 0.05)

### Statistical Properties
- PPI intervals have proven finite-sample coverage guarantees
- The method is distribution-free (makes minimal assumptions about data distribution)
- Intervals adapt to prediction quality automatically

### Computational Complexity
- PPI scales efficiently with dataset size
- Most computation is in the initial ML model training
- Inference step is fast and parallelizable

## Limitations & Considerations

1. **Prediction Quality**: PPI benefits require predictions that are better than random
2. **Labeled Data**: Still requires some labeled data for bias correction
3. **Model Assumptions**: Assumes predictions are available for the target population
4. **Implementation**: Requires careful handling of data splits to avoid leakage

## Related Work

Prediction-Powered Inference was introduced in:
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023). "Prediction-powered inference." *arXiv preprint arXiv:2301.09633*.

## Contributing

Feel free to:
- Report issues with the notebook
- Suggest improvements to the analysis
- Add new comparison methods
- Improve documentation

## License

This project is for educational and research purposes. Please cite the original PPI paper if using this methodology in academic work.