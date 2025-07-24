# Machine Learning Project

A comprehensive machine learning project implementing traditional algorithms for both numerical regression and image classification tasks. Each notebook compares two different models to evaluate performance.

## Project Structure

```
├── numericalDS.ipynb          # Car Price Prediction
├── catsVSdogs.ipynb          # Binary Image Classification  
├── breeds.ipynb              # Multi-class Image Classification
└── README.md                 # Documentation
```

## Projects Overview

### 1. Car Price Prediction (`numericalDS.ipynb`)
Predicts pre-owned car prices using vehicle specifications.

**Models**: Linear Regression | KNN Regressor  
**Dataset**: Car features (brand, transmission, fuel type, engine capacity, etc.)

**Results**:
- **Linear Regression**: 96% accuracy, MSE: 0.047, R²: 0.955
- **KNN Regressor**: 90% accuracy, MSE: 0.101, R²: 0.904

![Linear Regression Results](images/lr_actual_vs_predicted.png)
<img width="588" height="457" alt="image" src="https://github.com/user-attachments/assets/c7b9e7a3-75f2-4dd8-bf97-52cfbbf1ec3e" />

![KNN Residual Plot](images/knn_residual_plot.png)

![Price Distribution](images/price_distribution.png)

![Correlation Heatmap](images/correlation_heatmap.png)

### 2. Cats vs Dogs Classification (`catsVSdogs.ipynb`)
Binary classification of cat and dog images.

**Models**: Logistic Regression | KNN Classifier  
**Dataset**: Animal images (32x32 pixels, PCA: 100 components)

**Results**:
- **Logistic Regression**: 67.89% accuracy, AUC: 0.52
- **KNN Classifier**: 65.58% accuracy, AUC: 0.58

![Logistic Regression Confusion Matrix](images/cats_dogs_lr_confusion_matrix.png)

![Logistic Regression ROC](images/cats_dogs_lr_roc.png)

![KNN Confusion Matrix](images/cats_dogs_knn_confusion_matrix.png)

![KNN ROC](images/cats_dogs_knn_roc.png)

![Sample Predictions](images/cats_dogs_predictions.png)

### 3. Breed Classification (`breeds.ipynb`)
Multi-class classification of animal breeds (4 categories).

**Models**: Logistic Regression | KNN Classifier  
**Dataset**: Breed images - Abyssinian, Bombay, Birman, Samoyed (PCA: 300 components)

**Results**:
- **Logistic Regression**: 73.75% accuracy, Precision: 0.74, Recall: 0.74
- **KNN Classifier**: Precision: 0.60, Recall: 0.74

![Breeds LR Confusion Matrix](images/breeds_lr_confusion_matrix.png)

![Breeds LR AUC](images/breeds_lr_auc.png)

![Breeds KNN Confusion Matrix](images/breeds_knn_confusion_matrix.png)

![Breeds KNN AUC](images/breeds_knn_auc.png)

![Breed Predictions](images/breed_predictions.png)

## Performance Summary

| Project | Linear Regression/Logistic | KNN | Best Model |
|---------|---------------------------|-----|------------|
| Car Price Prediction | 96% (R²: 0.955) | 90% (R²: 0.904) | Linear Regression |
| Cats vs Dogs | 67.89% (AUC: 0.52) | 65.58% (AUC: 0.58) | Logistic Regression |
| Breed Classification | 73.75% | 60% Precision | Logistic Regression |

## Technologies

- **Python** - Core language
- **Scikit-learn** - ML algorithms & metrics  
- **OpenCV** - Image processing
- **Pandas/NumPy** - Data manipulation
- **Matplotlib/Seaborn** - Visualization

## Quick Start

### Installation
```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn opencv-python scipy
```

### Usage
1. Clone repository
2. Launch Jupyter: `jupyter notebook`  
3. Run desired notebook

### Save Visualizations
Add to your notebook cells:
```python
plt.savefig('images/filename.png', dpi=300, bbox_inches='tight')
```

## Key Features

- **Data Preprocessing**: Outlier removal, feature encoding, normalization
- **Feature Engineering**: PCA dimensionality reduction for images
- **Model Comparison**: Systematic evaluation of different algorithms
- **Comprehensive Metrics**: Accuracy, MSE, confusion matrices, ROC curves

## Future Work

- Ensemble methods implementation
- Advanced feature engineering
- Cross-validation integration
- Deep learning comparison

---

*This project demonstrates traditional ML algorithms applied to diverse datasets, showcasing fundamental techniques for regression and classification tasks.* 
