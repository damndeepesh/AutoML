# AutoML + Explainability Web App

This project is a Streamlit web application designed to democratize machine learning by allowing users to easily upload data, train models, compare their performance, understand model decisions through SHAP explanations, and export the best model.

## 🧭 Project Purpose

- Allow non-technical users, analysts, and data scientists to:
    - Upload structured data files (`.csv`, `.xlsx`).
    - Automatically train and compare various ML models.
    - View model performance metrics on a leaderboard.
    - Interpret model decisions using SHAP (SHapley Additive exPlanations).
    - Download the best-performing model for deployment or further analysis.

## 🧠 Key Concepts

- **AutoML (Automated Machine Learning):** Automates the end-to-end machine learning pipeline, including data preprocessing, model selection, and evaluation.
- **Model Explainability (XAI):** Uses SHAP to provide insights into how models make predictions, enhancing transparency and trust.

## 🧱 Functional Components

1.  **File Upload & Data Ingestion:** Supports `.csv` and `.xlsx` file uploads with data preview and column information.
2.  **Target Column Selection:** Users select the dependent variable for prediction.
3.  **Automated Model Training:** Handles preprocessing (imputation, encoding, scaling) and trains multiple classification or regression models (e.g., Logistic Regression, Random Forest, Gradient Boosting, SVM).
4.  **Leaderboard Display:** Ranks models based on performance metrics (Accuracy, F1-score, AUC for classification; R2 for regression).
5.  **Explainability Dashboard:** Visualizes feature importance and SHAP summary plots (beeswarm, waterfall) to explain model behavior.
6.  **Model Export Functionality:** Allows users to download the best model (along with any preprocessing steps like scalers) as a `.joblib` or `.pkl` pipeline.

## ⚙️ Setup and Installation

1.  **Clone the repository (or download the files):**
    ```bash
    # If you have git installed
    # git clone <repository_url>
    # cd AutoML-Explainability-WebApp
    ```
    Ensure `app.py`, `requirements.txt`, and this `README.md` are in your project directory (`/Users/damndeepesh/Documents/AutoML/`).

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Running the Application

1.  Navigate to the project directory in your terminal:
    ```bash
    cd /Users/damndeepesh/Documents/AutoML/
    ```

2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

3.  Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

## 🛠️ How to Use

1.  **Data Upload & Preview:** Upload your dataset and select the target column.
2.  **Model Training:** Configure training parameters (test size, CV folds) and click "Start Training".
3.  **Model Comparison:** View the leaderboard and performance visualizations.
4.  **Explainability:** Explore SHAP plots for the best model.
5.  **Model Export:** Download the trained model pipeline.

## ✨ Optional Advanced Features (Future Enhancements)

-   Model Upload Support
-   Live Prediction Interface
-   Confusion Matrix & ROC Curve visualizations
-   Extended support for Multiclass Classification and more Regression metrics
-   Data Cleaning Suggestions
-   Hyperparameter Tuning Panel

---
_This application structure was generated with assistance from an AI pair programmer._