# Rolling Correlations and Granger Causality-Driven LSTM Modeling for Stock Market Prediction: A Study of NASDAQ Interdependencies

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python version">
  <img src="https://img.shields.io/badge/status-active-brightgreen.svg" alt="Project status">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/notebooks-Jupyter-orange.svg" alt="Jupyter">
  <img src="https://img.shields.io/badge/finance-NASDAQ-blueviolet.svg" alt="Finance domain">
</p>


## Abstract

Financial markets are complex systems characterized by intricate interdependencies, where the price dynamics of one asset can significantly affect others. This project explores whether the future price of a NASDAQ-listed stock \( X \) can be predicted based solely on another stock \( Y \), their rolling time-window (RTW) correlation, and the assumption that \( Y \) Granger-causes \( X \).

We propose an end-to-end methodology that integrates:
- **Rolling Correlation Analysis**  
- **Granger Causality Testing**  
- **LSTM-based Deep Learning Forecasting**

The project serves both as a case study in financial interdependency modeling and a practical demonstration of combining statistical and deep learning techniques in a market forecasting context.

The full pipeline is implemented in the `main.ipynb` notebook inside the `src/` folder, from data loading and preprocessing to causality analysis, model training, and evaluation.

---

## Prerequisites

- Python (≥ 3.12.9)
- pip (Python package manager)
- Git (for handling submodules)
- Visual Studio Code (recommended)

### Recommended VS Code Extensions:
- Python (`ms-python.python`)
- Jupyter (`ms-toolsai.jupyter`)

---

## Installation

1. **Clone the repository (with submodules):**
   ```bash
   git clone --recursive https://github.com/blackswan-quants/nasdaq_causal-analysis_lstm.git
   cd nasdaq_causal-analysis_lstm
   ```

   If you forgot `--recursive`, you can run:
   ```bash
   git submodule update --init --recursive
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   ```

   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\.venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Configure VS Code:**
   Open the project folder in VS Code and select the `.venv` environment (Cmd/Ctrl+Shift+P → *Python: Select Interpreter*).

---

## Data

All required datasets are stored in the `data/pickle_files/` folder. The key files include:

- `cleaned_nasdaq_dataframe.pkl`: Preprocessed historical price data of NASDAQ stocks.
- Additional `.pkl` files representing derived datasets used across different stages of the pipeline.

Make sure all files in this folder are intact before running the analysis.

---

## Usage

To run the complete pipeline:

1. Complete all steps in the **Installation** section.
2. Open the project in VS Code and ensure the correct Python environment (`.venv`) is selected.
3. Launch the `main.ipynb` notebook inside the `src/` folder.
4. Run the following notebooks in order:
   - `exploratory_data_analysis.ipynb`  
   - `part1.ipynb` (Correlation + Granger Causality)  
   - `part2.ipynb` (LSTM Modeling with Causal Features)

Intermediate models, plots, and processed data will be saved to corresponding subfolders for reproducibility and inspection.

---

## Methodology Summary

1. **Exploratory Data Analysis:**  
   Understand trends, seasonality, missingness, and volatility in the NASDAQ stock time series.

2. **Part 1 – Correlation & Causality:**  
   - Compute rolling-window Pearson correlations between stocks.
   - Test for Granger causality between selected pairs using stationary transformations (ADF + differencing).

3. **Part 2 – Predictive Modeling:**  
   - Use LSTM networks to forecast target stock \( X \) using input features derived from stock \( Y \) and their rolling correlation.
   - Evaluate out-of-sample performance and assess how causality relationships improve predictability.

---

## Team & Contact

This project was developed by the **BlackSwan Quants** team.

- **Giulia Talà** – [LinkedIn](https://www.linkedin.com/in/giuliatala/)  
- **Gloria Desideri** – [LinkedIn](https://www.linkedin.com/in/gloria-desideri/)  
- **Simone Zani** – [LinkedIn](https://www.linkedin.com/in/simonezani35/)

For inquiries or collaboration opportunities, feel free to reach out via LinkedIn or GitHub.
