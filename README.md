# Machine Learning Project: Data Science and Image Classification

This repository contains a comprehensive machine learning project focusing on both numerical data analysis and image classification tasks. The project is divided into three main components, each implemented in separate Jupyter notebooks, with **two machine learning models** used in each project for comparison.

## Project Structure

```
├── numericalDS.ipynb          # Car Price Prediction (Regression Analysis)
├── catsVSdogs.ipynb          # Binary Image Classification (Cats vs Dogs)
├── breeds.ipynb              # Multi-class Classification (Animal Breeds)
└── README.md                 # Project Documentation
```

## Project Components

### 1. Numerical Data Science Analysis (`numericalDS.ipynb`)
**Dataset**: Pre-owned Cars Price Prediction
- **Objective**: Predict car prices based on features like brand, transmission, fuel type, engine capacity, etc.
- **Data preprocessing**: Outlier removal using IQR and Z-score methods, feature encoding
- **Models Used**: 
  - Linear Regression
  - K-Nearest Neighbors (KNN) Regressor

**Results**:
- **Linear Regression**: 
  - Accuracy: **96%**
  - Mean Squared Error: **0.047**
  - Mean Absolute Error: **0.164**
  - R² Score: **0.955**
- **KNN Regressor**: 
  - Accuracy: **90%**
  - Mean Squared Error: **0.101**
  - Mean Absolute Error: **0.224**
  - R² Score: **0.904**

### 2. Cats vs Dogs Classification (`catsVSdogs.ipynb`)
**Dataset**: Binary image classification with cats and dogs images
- **Objective**: Classify images as either cats (class 0) or dogs (class 1)
- **Image preprocessing**: 32x32 pixel resize, normalization, PCA feature extraction (100 components)
- **Models Used**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN) Classifier

**Results**:
- **Logistic Regression**:
  - Test Accuracy: **67.89%**
  - AUC Score: **0.52**
  - Includes confusion matrix visualization
- **KNN Classifier**:
  - Test Accuracy: **65.58%**
  - AUC Score: **0.58**
  - Includes confusion matrix and ROC curve visualizations

### 3. Breed Classification (`breeds.ipynb`)
**Dataset**: Multi-class classification of animal breeds
- **Categories**: Abyssinian, Bombay, Birman, Samoyed (4 classes)
- **Objective**: Classify images into specific breed categories
- **Image preprocessing**: 32x32 pixel resize, normalization, PCA feature extraction (300 components)
- **Models Used**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN) Classifier

**Results**:
- **Logistic Regression**:
  - Test Accuracy: **73.75%**
  - Log Loss: **1.2519**
  - Precision: **0.74**
  - Recall: **0.74**
- **KNN Classifier**:
  - Log Loss: **7.8372**
  - Precision: **0.60**
  - Recall: **0.74**
  - Includes AUC visualization for multi-class classification

## Key Features

- **Data Preprocessing**: Comprehensive data cleaning, outlier removal, and feature engineering
- **Visualization**: Rich visualizations using matplotlib and seaborn
- **Feature Engineering**: PCA for dimensionality reduction in image tasks
- **Model Comparison**: Two different algorithms used in each project for performance comparison
- **Performance Metrics**: Complete evaluation including accuracy, MSE, confusion matrices, ROC curves

## Technologies Used

- **Python**: Primary programming language
- **Jupyter Notebook**: Development environment
- **OpenCV**: Image processing and loading
- **NumPy & Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and metrics
- **SciPy**: Statistical analysis (Z-score calculations)

## Performance Summary

| Project | Model | Accuracy/Performance | Key Metrics |
|---------|-------|---------------------|-------------|
| Car Price Prediction | Linear Regression | 96% | MSE: 0.047, R²: 0.955 |
| Car Price Prediction | KNN Regressor | 90% | MSE: 0.101, R²: 0.904 |
| Cats vs Dogs | Logistic Regression | 67.89% | AUC: 0.52 |
| Cats vs Dogs | KNN Classifier | 65.58% | AUC: 0.58 |
| Breed Classification | Logistic Regression | 73.75% | Precision: 0.74, Recall: 0.74 |
| Breed Classification | KNN Classifier | - | Precision: 0.60, Recall: 0.74 |

## Results Visualizations

### 1. Car Price Prediction (numericalDS.ipynb)

#### Price Distribution Analysis
![Price Distribution](images/price_distribution.png)
*Distribution of car prices before and after outlier removal*

#### Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)
*Feature correlation matrix showing relationships between car attributes*

#### Model Performance Comparison
![Actual vs Predicted](images/actual_vs_predicted.png)
*Linear Regression: Actual vs Predicted prices scatter plot*

![Residual Plot](images/residual_plot.png)
*KNN Regressor: Residual plot showing prediction errors*

### 2. Cats vs Dogs Classification (catsVSdogs.ipynb)

#### Logistic Regression Results
![Logistic Regression Confusion Matrix](images/cats_dogs_lr_confusion_matrix.png)
*Confusion Matrix - Logistic Regression (67.89% Accuracy)*

![Logistic Regression ROC](images/cats_dogs_lr_roc.png)
*ROC Curve - Logistic Regression (AUC: 0.52)*

#### KNN Classifier Results
![KNN Confusion Matrix](images/cats_dogs_knn_confusion_matrix.png)
*Confusion Matrix - KNN Classifier (65.58% Accuracy)*

![KNN ROC](images/cats_dogs_knn_roc.png)
*ROC Curve - KNN Classifier (AUC: 0.58)*

#### Sample Predictions
![Sample Predictions](images/cats_dogs_predictions.png)
*Sample images with true vs predicted labels*

### 3. Breed Classification (breeds.ipynb)

#### Logistic Regression Results
![Breeds LR Confusion Matrix](images/breeds_lr_confusion_matrix.png)
*Confusion Matrix - Logistic Regression (73.75% Accuracy)*

![Breeds LR AUC](images/breeds_lr_auc.png)
*AUC Values per Class - Logistic Regression*

#### KNN Classifier Results
![Breeds KNN Confusion Matrix](images/breeds_knn_confusion_matrix.png)
*Confusion Matrix - KNN Classifier*

![Breeds KNN AUC](images/breeds_knn_auc.png)
*AUC Values per Class - KNN Classifier*

#### Sample Breed Predictions
![Breed Predictions](images/breed_predictions.png)
*Sample breed classification results with true vs predicted labels*

### How to Generate and Save These Images

To save the visualizations from your notebooks, add these code snippets to your notebooks:

```python
# Save plots in your notebooks
import matplotlib.pyplot as plt

# After creating any plot, add:
plt.savefig('images/plot_name.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Recommended image saving locations in your notebooks:**

1. **numericalDS.ipynb**: After each visualization cell, add:
   ```python
   plt.savefig('images/price_distribution.png', dpi=300, bbox_inches='tight')
   plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
   plt.savefig('images/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
   plt.savefig('images/residual_plot.png', dpi=300, bbox_inches='tight')
   ```

2. **catsVSdogs.ipynb**: After confusion matrix and ROC plots:
   ```python
   plt.savefig('images/cats_dogs_lr_confusion_matrix.png', dpi=300, bbox_inches='tight')
   plt.savefig('images/cats_dogs_lr_roc.png', dpi=300, bbox_inches='tight')
   plt.savefig('images/cats_dogs_knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
   plt.savefig('images/cats_dogs_knn_roc.png', dpi=300, bbox_inches='tight')
   ```

3. **breeds.ipynb**: After each visualization:
   ```python
   plt.savefig('images/breeds_lr_confusion_matrix.png', dpi=300, bbox_inches='tight')
   plt.savefig('images/breeds_lr_auc.png', dpi=300, bbox_inches='tight')
   plt.savefig('images/breeds_knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
   ```

Create an `images/` folder in your project directory to store all the visualization results.

## Visualizations Included

### numericalDS.ipynb
- Price distribution histograms
- Correlation heatmap
- Actual vs Predicted price scatter plots
- Residual plots for model evaluation

### catsVSdogs.ipynb
- Confusion matrices for both models
- ROC curves with AUC scores
- Sample predictions with true vs predicted labels

### breeds.ipynb
- Confusion matrices for multi-class classification
- AUC bar charts for each class
- Sample image predictions with labels

## Getting Started

### Prerequisites
```bash
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
pip install opencv-python scipy
```

### Running the Notebooks

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open any of the three notebooks:
   - `numericalDS.ipynb` for car price prediction analysis
   - `catsVSdogs.ipynb` for binary image classification
   - `breeds.ipynb` for multi-class breed classification

## Dataset Requirements

- **Car Dataset**: CSV file with car features (brand, transmission, fuel type, etc.)
- **Image Datasets**: Organized image folders with proper annotations
- **Cats vs Dogs**: Images labeled by species (1 for cats, 2 for dogs)
- **Breeds**: Images categorized by breed names

## Future Improvements

- Implement ensemble methods combining both models
- Add more sophisticated feature engineering techniques
- Experiment with different PCA component numbers
- Apply advanced image preprocessing techniques
- Add cross-validation for more robust evaluation

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Note**: This project demonstrates the application of traditional machine learning algorithms (Linear Regression, Logistic Regression, and KNN) to both numerical and image data, showcasing the versatility of these fundamental techniques across different domains. 
