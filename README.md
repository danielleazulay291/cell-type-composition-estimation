# Cell-Type Composition Estimation from Gene Expression Data

This project implements a genetic-algorithm–based pipeline for estimating cell-type composition from bulk gene expression data.

The system models gene expression as a mixture of reference cell-type profiles and optimizes cell-type proportions under biological constraints, including non-negativity and normalization.

## Usage
```bash
python main.py

## Data
- Reference gene expression matrix representing known cell-type profiles
- Bulk gene expression samples

## Output
Estimated cell-type proportions per sample.

## Applications
Applicable to medical and biological data research, including tissue analysis, disease characterization, and exploratory research.

## Requirements
- Python 3
- numpy
- pandas
