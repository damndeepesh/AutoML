# AutoML & Explainability Web Application

This Streamlit web application empowers users to perform end-to-end machine learning tasks with ease. Upload your data, automatically train and compare various models, understand their predictions through SHAP explainability, and export the best model for your needs.

## 🎯 Core Objectives

*   **Accessibility**: Enable users of all technical backgrounds to leverage machine learning.
*   **Automation**: Streamline the ML pipeline from data ingestion to model evaluation.
*   **Transparency**: Provide clear insights into model behavior using SHAP.
*   **Efficiency**: Quickly identify the best-performing model for a given dataset.

## ✨ Key Features

*   **Flexible Data Upload**: 
    *   Supports `.csv` and `.xlsx` files.
    *   Option to upload a single file (for automatic train/test splitting) or separate training and testing files.
*   **Data Preprocessing**: 
    *   Automatic handling of missing values (imputation).
    *   Encoding of categorical features.
    *   Optional scaling of numeric features.
*   **Target Column & Problem Type Detection**: 
    *   Easy selection of the target variable.
    *   Automatic detection of problem type (Classification/Regression).
    *   Auto-detection of common target column names.
*   **Automated Model Training & Comparison**: 
    *   Trains a suite of models tailored to the problem type:
        *   **Classification**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, K-Nearest Neighbors, Gaussian Naive Bayes.
        *   **Regression**: Linear Regression, Ridge Regression, ElasticNet, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, SVR, K-Nearest Neighbors Regressor.
    *   Displays a leaderboard with key performance metrics (Accuracy, F1, AUC for classification; R2, MSE for regression).
*   **Model Explainability (XAI)**: 
    *   Utilizes SHAP (SHapley Additive exPlanations) for the best model.
    *   Global feature importance plots.
    *   Detailed SHAP summary plots (e.g., beeswarm) and individual prediction explanations (waterfall plots coming soon).
*   **Model Export**: Download the trained best model (including preprocessing steps) as a `.joblib` file for deployment or further use.

## ⚙️ Setup & Installation

1.  **Prerequisites**: Python 3.7+ installed.
2.  **Clone the Repository (Optional)**:
    ```bash
    # git clone <your_repository_url> # If you have it on Git
    # cd AutoML-WebApp
    ```
    Alternatively, ensure `app.py` and `requirements.txt` are in your project directory.
3.  **Create and Activate Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate    # Windows
    ```
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Running the Application

1.  Navigate to your project directory in the terminal.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  Open your browser and go to the URL provided (usually `http://localhost:8501`).

## 🔮 Upcoming Features & Enhancements

We are continuously working to improve this AutoML application. Here are some features on our roadmap:

*   **Advanced Preprocessing Options**: 
    *   User control over imputation strategies (mean, median, mode, constant).
    *   More encoding techniques (e.g., One-Hot Encoding, Target Encoding).
    *   Feature selection techniques.
*   **Hyperparameter Tuning**: 
    *   Integration of GridSearchCV or RandomizedSearchCV for optimizing model hyperparameters.
    *   User interface to define search spaces.
*   **Expanded Model Support**: 
    *   LightGBM, XGBoost, CatBoost for both classification and regression.
    *   Basic Time Series forecasting models (e.g., ARIMA, Prophet) if applicable data is provided.
*   **Enhanced Evaluation & Visualization**: 
    *   Interactive Confusion Matrix, ROC/AUC curves, Precision-Recall curves for classification.
    *   Residual plots, Actual vs. Predicted plots for regression.
    *   Cross-validation score details.
*   **Deployment & Integration**: 
    *   Option to generate a simple Flask API endpoint for the exported model.
    *   Dockerization support for easier deployment.
*   **User Experience & Robustness**: 
    *   More detailed error handling and user guidance.
    *   Saving and loading of experiment configurations.
    *   Support for larger datasets (optimizations for memory and speed).
*   **Advanced Explainability**: 
    *   Individual prediction explanations (waterfall plots).
    *   Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots.
*   **Data Insights**: 
    *   Automated exploratory data analysis (EDA) report generation.

---
_This application is actively developed, with assistance from AI pair programming._