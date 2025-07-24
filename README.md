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

- <img width="567" height="457" alt="image" src="https://github.com/user-attachments/assets/379f8ccd-55ec-40df-ac9e-05f75e8b6ea7" />

- **KNN Regressor**: 90% accuracy, MSE: 0.101, R²: 0.904
  
- <img width="567" height="457" alt="image" src="https://github.com/user-attachments/assets/1d44a5ed-0a32-425e-bfc4-01217b94976b" />




### 2. Cats vs Dogs Classification (`catsVSdogs.ipynb`)
Binary classification of cat and dog images.

**Models**: Logistic Regression | KNN Classifier  
**Dataset**: Animal images (32x32 pixels, PCA: 100 components)

**Results**:
- **Logistic Regression**: 67.89% accuracy, AUC: 0.52
  
- <img width="424" height="374" alt="image" src="https://github.com/user-attachments/assets/fac3b397-df35-41df-ae7d-48563c39af27" />

- **KNN Classifier**: 65.58% accuracy, AUC: 0.58
  
- <img width="424" height="374" alt="image" src="https://github.com/user-attachments/assets/1c01091c-918a-483d-a65a-cc60596ee046" />




### 3. Breed Classification (`breeds.ipynb`)
Multi-class classification of animal breeds (4 categories).


**Models**: Logistic Regression | KNN Classifier  
**Dataset**: Breed images - Abyssinian, Bombay, Birman, Samoyed (PCA: 300 components)

**Results**:
- **Logistic Regression**: 73.75% accuracy, Precision: 0.74, Recall: 0.74
- <img width="481" height="374" alt="image" src="https://github.com/user-attachments/assets/9c6ab730-5159-4258-bb44-1d51d421d878" />

- <img width="461" height="451" alt="image" src="https://github.com/user-attachments/assets/6b218d4a-6157-4f60-aa07-2b93beebe0a9" />

- **KNN Classifier**: Precision: 0.60, Recall: 0.74
- <img width="481" height="374" alt="image" src="https://github.com/user-attachments/assets/38d69302-cb08-4805-ba22-913809fad012" />

- <img width="461" height="451" alt="image" src="https://github.com/user-attachments/assets/8790316b-d226-4853-abd8-f201b4b5efd4" />




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
