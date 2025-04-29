# Helper Modules

This repository contains a collection of Python modules developed by the BlackswanQuants team to support various tasks in financial data analysis, machine learning, and quantitative research.  
It is intended to be used as a Git submodule within other projects.

## Overview

The `helpermodules` repository includes scripts and utilities for:

- Time series analysis (e.g., Dynamic Time Warping clustering, Granger causality)
- Deep learning models (e.g., LSTM implementations)
- Data retrieval and preprocessing
- Statistical analysis
- Portfolio visualization and risk assessment
- Text processing

## Repository Structure

```
helpermodules/
├── data/                      # Sample datasets and data-related resources
├── docs/                      # Documentation and usage examples
├── clustering_class.py        #DTW, Eucliean distance computation for k-means, hierarchical clustering functions
├── LSTM.py                    # Long Short-Term Memory model implementations
├── correlation_study.py       # Tools for correlation analysis
├── df_dataretrieval.py        # Data retrieval utilities
├── granger_causality.py       # Granger causality analysis functions
├── inflation.py               # Inflation data processing tools
├── memory_handling.py         # Memory management utilities
├── nonlin_granger_causality.py# Non-linear Granger causality analysis
├── portvis_assets.py          # Portfolio assets visualization tools
├── portvis_helpers.py         # Helper functions for portfolio visualization
├── portvis_risk.py            # Portfolio risk assessment tools
├── split_text_by_minute.py    # Text processing utilities
├── statistical_analysis.py    # Statistical analysis functions
└── requirements.txt           # List of required Python packages
```

## Installation

To use this repository as a submodule in your project:

```bash
git submodule add https://github.com/blackswan-quants/helpermodules.git helpermodules
git submodule update --init --recursive
```

Install the required Python packages:

```bash
pip install -r helpermodules/requirements.txt
```

## Usage

Import the modules in your Python scripts as needed. For example:

```python
from helpermodules.clustering_class import dtw_cluster
from helpermodules.LSTM import train_lstm_model
from helpermodules.df_dataretrieval import fetch_financial_data
```

Refer to the `docs/` directory for detailed usage examples and documentation.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact the BlackswanQuants team at [blackswan.quants@gmail.com](mailto:blackswan.quants@gmail.com).
