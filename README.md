# Identifying Key Policy Drivers of FLFP Growth: A Machine Learning Approach

This repository contains a modular machine learning pipeline developed for the Data Science and Advanced Programming 2025 course at UNIL. The project identifies the economic and legal drivers of Female Labour Force Participation (FLFP) growth across a global panel (1991–2021).

Quick Start: Running the Pipeline
To ensure scientific integrity and prevent data leakage, this project uses a strict temporal validation strategy (Training: 1996–2006; Testing: 2011–2016). Follow these steps to reproduce the results in a clean environment:

1. Clone and Navigate

git clone https://github.com/szahid5/DSAP_Proj_FLFPGrowth
cd DSAP_Proj_FLFPGrowth

2. Environment Setup

Create the reproducible Conda environment using the provided environment.yml.

conda env create -f environment.yml
conda activate flfp_project

3. Execute the Orchestrator

Run the central entry point to trigger the end-to-end analytical loop.
python main.py


Project Architecture
The project follows a modular design pattern to separate concerns and ensure maintainability:
main.py: The central orchestrator for the pipeline.
src/data_loader.py: Handles the ingestion of 9 datasets, ISO-3 code standardization, and 5-year lag engineering for legal indicators.
src/models.py: Contains the temporal validation split logic and scikit-learn pipeline construction.
src/evaluation.py: Generates diagnostic plots and calculates residuals to identify "Miracle Countries"

Key Outputs
After execution, all results are programmatically saved to the /results directory:
miracle_countries.csv: Automated identification of outliers like Saudi Arabia (residual of +0.45).
Visualizations: Includes Ridge coefficients, Partial Dependence plots (illustrating the Convergence Effect), and Reality Check scatter plots.

Maintenance and Updates
The modular structure allows for independent updates. For instance, data_loader.py can be refreshed with new World Bank annual releases without modifying the core training logic in models.py.


Contact: Samia Aisha Zahid (samia.zahid@unil.ch).