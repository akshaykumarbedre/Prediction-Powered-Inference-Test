# Quick Setup Guide

This document provides step-by-step instructions to get the Prediction-Powered Inference test up and running.

## Prerequisites

- Python 3.7 or higher
- pip package manager
- Git (for cloning the repository)

## Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone https://github.com/akshaykumarbedre/Prediction-Powered-Inference-Test
cd Prediction-Powered-Inference-Test
```

### 2. Install Standard Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install PPI Library

The `ppi_py` library may not be available via standard pip. Try these options:

#### Option A: Try pip first
```bash
pip install ppi_py
```

#### Option B: Install from source (if Option A fails)
```bash
git clone https://github.com/aangelopoulos/ppi_py
cd ppi_py
pip install -e .
cd ..
```

#### Option C: Alternative PPI implementation
If the above doesn't work, you can install alternative implementations:
```bash
pip install ppi-python
```

### 4. Verify Installation
Test that all imports work:
```python
python -c "
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
print('Basic dependencies OK')

try:
    import ppi_py
    print('ppi_py imported successfully')
except ImportError:
    print('ppi_py not available - see README for alternatives')

from utils import make_plots
print('Utils module OK')
"
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook ppi_test_on_fetch_california_housing.ipynb
```

## Troubleshooting

### PPI Library Issues
If you can't install `ppi_py`, you can:
1. Comment out the PPI-related imports in the notebook
2. Focus on the Classical and Imputation methods comparison
3. Use the notebook structure as a template for other PPI implementations

### Missing Utils
The `utils.py` file is included in this repository and contains the plotting functions needed for visualization.

### Python Version Compatibility
If you encounter issues, ensure you're using Python 3.7+:
```bash
python --version
```

### Virtual Environment (Recommended)
For clean installation, use a virtual environment:
```bash
python -m venv ppi_env
source ppi_env/bin/activate  # On Windows: ppi_env\Scripts\activate
pip install -r requirements.txt
```

## Quick Test

To verify everything is working, run this minimal test:
```python
from sklearn.datasets import fetch_california_housing
import numpy as np

# Load data
housing = fetch_california_housing()
print(f"Loaded {housing.data.shape[0]} samples with {housing.data.shape[1]} features")
print("Setup successful!")
```

## Next Steps

Once setup is complete:
1. Open the Jupyter notebook
2. Run cells sequentially
3. Examine the generated visualizations
4. Experiment with different parameters
5. Try applying PPI to your own datasets

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify Python version compatibility
3. Review the error messages carefully
4. Consult the main README.md for more detailed information