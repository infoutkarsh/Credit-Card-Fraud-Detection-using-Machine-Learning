# Credit Card Fraud Detection

## Project Overview
Credit card fraud detection is a critical task to prevent financial losses and secure transactions in the banking and financial domain. This project evaluates and compares the performance of multiple machine learning models to classify fraudulent and legitimate transactions effectively.

## Features and Dataset
### Dataset:
The dataset used contains anonymized transaction records, with a "Class" label indicating fraudulent (1) and non-fraudulent (0) transactions.

### Features:
- **Time**: Time elapsed since the first transaction in the dataset.
- **Amount**: Transaction amount.
- **V1 to V28**: Principal Component Analysis (PCA) transformed features to protect sensitive information.
- **Class**: Target variable (1: Fraud, 0: Legitimate).

## Objectives
- Preprocess the dataset to handle class imbalance and outliers.
- Train multiple machine learning models to predict fraudulent transactions.
- Evaluate and compare model performances using key metrics like F1-Score, AUC-ROC, Precision, and Recall.

## Machine Learning Models Used
1. **Random Forest**  
   - Ensemble method with decision trees to handle complex relationships in data.
   - **Key Metrics**:  
     - AUC-ROC: 0.908053  
     - Precision: 0.948718  
     - Recall: 0.816176  
     - F1-Score: 0.87747  

2. **Decision Tree**  
   - Single tree-based model for classification tasks.

3. **Support Vector Machine (SVM)**  
   - Finds the hyperplane that best separates classes in high-dimensional spaces.

4. **Neural Networks**  
   - Multi-layer perceptron for learning non-linear relationships in data.

5. **Logistic Regression**  
   - Simple linear model for binary classification tasks.

## Steps in Implementation
1. **Data Preprocessing**  
   - Handle missing values and scale features.
   - Address class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

2. **Model Training**  
   - Split data into training and testing sets (80/20 split).
   - Train each model with hyperparameter tuning using GridSearchCV.

3. **Evaluation Metrics**  
   - **AUC-ROC**: Measures the model's ability to differentiate between classes.  
   - **Precision**: Proportion of true positives among predicted positives.  
   - **Recall**: Proportion of true positives among actual positives.  
   - **F1-Score**: Harmonic mean of Precision and Recall for balanced performance.

## Results and Conclusion
### Model Comparison:
| Model               | AUC-ROC  | Precision  | Recall   | F1-Score  |
|---------------------|----------|------------|----------|-----------|
| **Random Forest**    | **0.908053** | **0.948718** | **0.816176** | **0.87747** |
| Decision Tree       | 0.872015 | 0.891472   | 0.778643 | 0.831876  |
| SVM                 | 0.863419 | 0.899317   | 0.759184 | 0.823833  |
| Neural Networks     | 0.880256 | 0.921477   | 0.784103 | 0.847987  |
| Logistic Regression | 0.854219 | 0.876923   | 0.726315 | 0.794872  |

### Conclusion:
The **Random Forest model** is identified as the best performer based on the F1-Score metric. It achieves a balanced performance with an **AUC-ROC of 0.908053**, **Precision of 0.948718**, **Recall of 0.816176**, and **F1-Score of 0.87747**, making it a robust choice for detecting credit card fraud in the given dataset.

---

## Installation and Usage
### Prerequisites:
- Python 3.7+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`

### Setup:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/credit-card-fraud-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python fraud_detection.py
   ```

## Future Work
- Test on real-world datasets for broader applicability.
- Explore advanced ensemble techniques like XGBoost and CatBoost.
- Integrate with streaming platforms for real-time fraud detection.
