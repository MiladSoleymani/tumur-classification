# Tumor Classification Using Machine Learning

A comprehensive machine learning project for tumor subtype classification using mutation and copy number variation (CNV) data.

## Overview

This repository contains various machine learning approaches to classify tumor subtypes (PDM vs SCM) using genomic features including:
- Tumor Mutation Burden (TMB)
- Percentage of Genome Altered (PGA)
- Missense mutations across 140 genes
- Copy number variations (logCR values) across 516 genes

## Dataset

The dataset (`data/combined_mutation_CNV.csv`) contains:
- **43 samples** from two tumor subtypes
- **656 features** including mutation and CNV data
- **2 target classes**: PDM and SCM

### Features:
- TMB (Tumor Mutation Burden)
- PGA (Percentage of Genome Altered)
- 140 binary missense mutation features
- 516 continuous copy number variation features (logCR values)

## Project Structure

```
tumur-classification/
├── data/
│   └── combined_mutation_CNV.csv          # Main dataset
├── jupyter/
│   ├── data_describtion.ipynb             # Data exploration and visualization
│   ├── classic_ml_models.ipynb            # Traditional ML algorithms
│   ├── DNN.ipynb                          # Deep Neural Network implementation
│   ├── PCA.ipynb                          # Principal Component Analysis
│   ├── classic_ml_models_using_PCA.ipynb  # ML models with PCA
│   ├── Permutation_feature_importance.ipynb # Feature importance analysis
│   ├── data_augmentation.ipynb            # Data augmentation techniques
│   ├── few_shot_learning.ipynb            # Few-shot learning approaches
│   ├── plots.ipynb                        # Visualization and plots
│   │
│   ├── classic_ml_models_GWO_feature_selection.ipynb  # Grey Wolf Optimizer
│   ├── classic_ml_models_PSO_feature_selection.ipynb  # Particle Swarm Optimization
│   ├── classic_ml_models_WOA_feature_selection.ipynb  # Whale Optimization Algorithm
│   │
│   ├── classic_ml_models_GWO_feature_selection_and_PCA.ipynb  # GWO + PCA
│   ├── classic_ml_models_PSO_feature_selection_and_PCA.ipynb  # PSO + PCA
│   ├── classic_ml_models_WOA_feature_selection_and_PCA.ipynb  # WOA + PCA
│   │
│   └── results/                           # Experimental results
│       ├── ML.csv                         # Classic ML results
│       ├── ML_PCA.csv                     # ML with PCA results
│       ├── ML_GOW.csv                     # ML with GWO results
│       ├── ML_PSO.csv                     # ML with PSO results
│       ├── ML_WOA.csv                     # ML with WOA results
│       └── ...                            # Other result files
└── README.md
```

## Implemented Approaches

### 1. Classical Machine Learning Models
- Support Vector Machines (SVM)
- k-Nearest Neighbors (k-NN)
- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost

### 2. Feature Selection Methods
- **Grey Wolf Optimizer (GWO)** - Bio-inspired optimization for feature selection
- **Particle Swarm Optimization (PSO)** - Swarm intelligence-based feature selection
- **Whale Optimization Algorithm (WOA)** - Nature-inspired optimization technique

### 3. Dimensionality Reduction
- Principal Component Analysis (PCA)
- Combined approaches (Feature Selection + PCA)

### 4. Advanced Techniques
- Deep Neural Networks (DNN)
- Few-shot Learning
- Data Augmentation
- Permutation Feature Importance

## Key Results

Based on the classical ML models without feature selection:
- **Logistic Regression**: 69.17% accuracy
- **SVM**: 66.94% accuracy
- **XGBoost**: 65.28% accuracy
- **k-NN**: 64.44% accuracy
- **Decision Trees**: 51.94% accuracy

*Note: Results with feature selection and PCA are available in the `jupyter/results/` directory.*

## Methodology

1. **Data Preprocessing**: 
   - MinMax scaling (range: -1 to 1)
   - Binary encoding for subtypes (PDM=0, SCM=1)

2. **Model Evaluation**:
   - 5-fold stratified cross-validation
   - Grid search for hyperparameter optimization
   - Metrics: Accuracy, Precision, Recall, F1-score

3. **Feature Engineering**:
   - Metaheuristic optimization algorithms for feature selection
   - PCA for dimensionality reduction
   - Permutation importance for feature ranking

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tumur-classification.git
cd tumur-classification
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

3. Run Jupyter notebooks:
```bash
cd jupyter
jupyter notebook
```

4. Start with `data_describtion.ipynb` for data exploration, then proceed to other notebooks based on your interest.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- jupyter

## Future Work

- Implement ensemble methods combining multiple feature selection techniques
- Explore deep learning architectures for genomic data
- Investigate explainable AI techniques for feature interpretation
- Extend to multi-class classification with additional tumor subtypes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please open an issue in this repository.