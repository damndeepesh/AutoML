import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import base64
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AutoML + Explainability Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.success-message {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #c3e6cb;
}
.stButton>button {
    width: 100%;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_model_metrics(y_true, y_pred, y_proba=None, problem_type='Classification'):
    metrics = {}
    if problem_type == "Classification":
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['F1-score'] = f1_score(y_true, y_pred, average='weighted')
        if y_proba is not None and len(np.unique(y_true)) == 2: # AUC for binary classification
            try:
                metrics['AUC'] = roc_auc_score(y_true, y_proba[:, 1])
            except ValueError:
                metrics['AUC'] = None # Handle cases where AUC cannot be computed
        else:
            metrics['AUC'] = None
    elif problem_type == "Regression":
        from sklearn.metrics import r2_score, mean_squared_error
        metrics['R2'] = r2_score(y_true, y_pred)
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        # Add other regression metrics if desired, e.g., MAE
    return metrics

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'data': None, 'target_column': None, 'problem_type': None,
        'models': {}, 'model_scores': {}, 'best_model_info': None,
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
        'le_dict': {}, 'scaler': None, 'trained_pipeline': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Page Functions ---
def data_upload_page():
    st.header("📁 Data Upload & Preview")

    upload_option = st.radio(
        "Select data upload method:",
        ('Single File (auto-split train/test)', 'Separate Train and Test Files'),
        key='upload_option'
    )

    uploaded_file = None
    uploaded_train_file = None
    uploaded_test_file = None

    if upload_option == 'Single File (auto-split train/test)':
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset. It will be split into training and testing sets.",
            key='single_file_uploader'
        )
    else:
        uploaded_train_file = st.file_uploader(
            "Choose a Training CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your training dataset.",
            key='train_file_uploader'
        )
        uploaded_test_file = st.file_uploader(
            "Choose a Testing CSV or Excel file (Optional)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your testing dataset. If not provided, the training data will be split.",
            key='test_file_uploader'
        )

    df = None
    df_train = None
    df_test = None

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.session_state.train_data = None # Clear separate train/test if single is uploaded
            st.session_state.test_data = None
            st.session_state.target_column = None
            st.session_state.problem_type = None
            st.session_state.source_data_type = 'single'
        except Exception as e:
            st.error(f"Error reading single file: {e}")
            return
    elif uploaded_train_file:
        try:
            df_train = pd.read_csv(uploaded_train_file) if uploaded_train_file.name.endswith('.csv') else pd.read_excel(uploaded_train_file)
            st.session_state.train_data = df_train
            st.session_state.data = df_train # Use train_data as primary for column selection initially
            df = df_train # for common processing below
            st.session_state.target_column = None
            st.session_state.problem_type = None
            st.session_state.source_data_type = 'separate'
            if uploaded_test_file:
                df_test = pd.read_csv(uploaded_test_file) if uploaded_test_file.name.endswith('.csv') else pd.read_excel(uploaded_test_file)
                st.session_state.test_data = df_test
            else:
                st.session_state.test_data = None # Explicitly set to None
        except Exception as e:
            st.error(f"Error reading train/test files: {e}")
            return

    if df is not None:
        try:
            # Common processing for df (either single or train_df)
            st.subheader("Data Overview" + (" (Training Data)" if st.session_state.get('source_data_type') == 'separate' else ""))

            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())

            st.subheader("Data Preview (First 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Column Information")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            }).reset_index(drop=True)
            st.dataframe(info_df, use_container_width=True)

            st.subheader("🎯 Target Column Selection")
            common_target_names = ['target', 'Target', 'label', 'Label', 'class', 'Class', 'Output', 'output', 'result', 'Result']
            detected_target = None
            df_columns = df.columns.tolist()
            for col_name in common_target_names:
                if col_name in df_columns:
                    detected_target = col_name
                    break
            
            target_options = [None] + df_columns
            target_index = 0
            if detected_target:
                try:
                    target_index = target_options.index(detected_target)
                except ValueError:
                    target_index = 0 # Should not happen if detected_target is in df_columns

            target_column = st.selectbox(
                "Select the target column (what you want to predict):",
                options=target_options,
                index=target_index,
                help="Choose the dependent variable. Common names are auto-detected."
            )

            auto_run_training = st.checkbox("Automatically start training when target is selected/detected?", value=False, key='auto_run_cb')

            if target_column:
                st.session_state.target_column = target_column
                target_series = df[target_column]
                
                # Determine problem type
                if target_series.nunique() <= 2 or (target_series.dtype == 'object' and target_series.nunique() <=10) :
                    st.session_state.problem_type = "Classification"
                    if target_series.dtype == 'object':
                        le = LabelEncoder()
                        df[target_column] = le.fit_transform(target_series)
                        st.session_state.le_dict[target_column] = le # Store encoder for target
                elif pd.api.types.is_numeric_dtype(target_series):
                    st.session_state.problem_type = "Regression"
                else:
                    st.session_state.problem_type = "Unsupported Target Type"
                    st.error("Target column type is not suitable for classification or regression.")
                    return

                st.success(f"Target column '{target_column}' selected. Problem Type: {st.session_state.problem_type}")

                if st.session_state.get('source_data_type') == 'separate' and st.session_state.test_data is not None:
                    st.subheader("Test Data Overview")
                    col1_test, col2_test, col3_test = st.columns(3)
                    col1_test.metric("Test Rows", st.session_state.test_data.shape[0])
                    col2_test.metric("Test Columns", st.session_state.test_data.shape[1])
                    col3_test.metric("Test Missing Values", st.session_state.test_data.isnull().sum().sum())
                    st.dataframe(st.session_state.test_data.head(5), use_container_width=True)
                    if target_column not in st.session_state.test_data.columns:
                        st.error(f"Target column '{target_column}' not found in the uploaded test data. Please ensure column names match.")
                        return # Stop further processing if target is missing in test data

                st.subheader(f"Target Column Distribution (in {'Training Data' if st.session_state.get('source_data_type') == 'separate' else 'Uploaded Data'}): {target_column}")
                if st.session_state.problem_type == "Classification":
                    fig, ax = plt.subplots()
                    sns.countplot(x=target_series, ax=ax)
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots()
                    sns.histplot(target_series, kde=True, ax=ax)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error reading or processing file: {e}")
            if auto_run_training and st.session_state.target_column:
                st.session_state.auto_run_triggered = True
                st.experimental_rerun() # Rerun to switch page or trigger training

        except Exception as e:
            st.error(f"Error processing data: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("👆 Please upload a CSV or Excel file (or separate train/test files) to get started.")

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Impute missing values in target variable y
    if y.isnull().any():
        if st.session_state.problem_type == "Classification":
            # For classification, ensure y is int/str before mode imputation if it's float with NaNs
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 2: # Check if it might be a float target for classification
                 # If it's float and intended for classification, it might have been label encoded already or needs specific handling.
                 # For now, let's assume if it's numeric and classification, it's likely already encoded or will be handled by LabelEncoder later.
                 # If it's float due to NaNs, mode might be tricky. Let's ensure it's treated as object for mode for safety.
                 y_imputer = SimpleImputer(strategy='most_frequent')
                 y[:] = y_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
            else:
                y_imputer = SimpleImputer(strategy='most_frequent')
                y[:] = y_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
        elif st.session_state.problem_type == "Regression":
            y_imputer = SimpleImputer(strategy='mean')
            y[:] = y_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
        st.warning(f"NaN values found and imputed in the target column '{target_column}'.")

    # Impute missing values in features X
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include='object').columns

    if len(num_cols) > 0:
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
    if len(cat_cols) > 0:
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # Encode categorical features
    le_dict_features = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict_features[col] = le
    st.session_state.le_dict.update(le_dict_features)

    # Ensure target y is correctly typed after imputation, especially for classification
    if st.session_state.problem_type == "Classification" and target_column in st.session_state.le_dict:
        # If target was label encoded, ensure it's integer type after imputation
        # This might be redundant if LabelEncoder was applied after imputation, but good for safety
        pass # y should already be encoded if it was object type initially
    elif st.session_state.problem_type == "Classification" and y.dtype == 'float':
        # If y is float after mean imputation (e.g. binary 0/1 became float)
        # and it's for classification, convert to int if appropriate
        # This case should be rare if 'most_frequent' is used for classification target imputation
        # However, if it was numeric and became float due to NaNs, then imputed with mean (which is wrong for classification)
        # This indicates a logic flaw in imputation strategy selection above. Assuming 'most_frequent' was used.
        pass 

    return X, y

def model_training_page():
    st.header("🚀 Model Training")
    # Check if data is available from either single upload or separate train/test upload
    data_available = (st.session_state.data is not None) or \
                     (st.session_state.train_data is not None)
    if not data_available or st.session_state.target_column is None:
        st.warning("⚠️ Please upload data (single or train/test) and select a target column first.")
        return
    if st.session_state.problem_type == "Unsupported Target Type":
        st.error("Cannot train models with the current target column type.")
        return

    target = st.session_state.target_column

    st.subheader("Training Configuration")
    col1, col2 = st.columns(2)
    # Disable test_size slider if separate test data is provided
    disable_test_size = st.session_state.get('source_data_type') == 'separate' and st.session_state.test_data is not None
    test_size = col1.slider("Test Size (if splitting single file)", 0.1, 0.5, 0.2, 0.05, disabled=disable_test_size)
    random_state = col1.number_input("Random State", value=42, min_value=0)
    cv_folds = col2.slider("Cross-Validation Folds", 3, 10, 5)
    scale_features = col2.checkbox("Scale Numeric Features", value=True)

    # Auto-start training if triggered
    start_button_pressed = st.button("🎯 Start Training", type="primary", key='manual_start_train_button')
    if st.session_state.get('auto_run_triggered_for_training') and not start_button_pressed:
        st.session_state.auto_run_triggered_for_training = False # Reset trigger
        start_button_pressed = True # Simulate button press
        st.info("🤖 Auto-training initiated...")

    if start_button_pressed:
        with st.spinner("Preprocessing data and training models..."):
            try:
                X_train, X_test, y_train, y_test = None, None, None, None
                
                if st.session_state.get('source_data_type') == 'separate' and st.session_state.train_data is not None:
                    df_train_processed = st.session_state.train_data.copy()
                    X_train, y_train = preprocess_data(df_train_processed, target)

                    if st.session_state.test_data is not None:
                        df_test_processed = st.session_state.test_data.copy()
                        if target not in df_test_processed.columns:
                            st.error(f"Target column '{target}' not found in test data during preprocessing. Aborting.")
                            return
                        X_test, y_test = preprocess_data(df_test_processed, target) # Preprocess test data separately
                        # Ensure X_test has same columns as X_train after preprocessing (esp. after one-hot encoding if added later)
                        # For now, LabelEncoder is per-column, SimpleImputer fits on data it sees.
                        # If one-hot encoding is added, fit on X_train, transform X_test, align columns.
                    else: # No test file, split train_data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_train, y_train, test_size=test_size, random_state=random_state, 
                            stratify=(y_train if st.session_state.problem_type == "Classification" else None)
                        )
                else: # Single file upload
                    df_processed = st.session_state.data.copy()
                    X, y = preprocess_data(df_processed, target)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, 
                        stratify=(y if st.session_state.problem_type == "Classification" else None)
                    )

                if X_train is None or y_train is None:
                    st.error("Training data (X_train, y_train) could not be prepared. Please check your data and selections.")
                    return

                # Scaling should be fit on X_train and transformed on X_test
                if scale_features:
                    num_cols_train = X_train.select_dtypes(include=np.number).columns
                    if len(num_cols_train) > 0:
                        scaler = StandardScaler()
                        X_train[num_cols_train] = scaler.fit_transform(X_train[num_cols_train])
                        st.session_state.scaler = scaler # Save the fitted scaler
                        if X_test is not None:
                            num_cols_test = X_test.select_dtypes(include=np.number).columns
                            # Ensure test set uses the same numeric columns in the same order as train set for scaling
                            cols_to_scale_in_test = [col for col in num_cols_train if col in X_test.columns]
                            if len(cols_to_scale_in_test) > 0:
                                # Create a DataFrame with columns in the order of num_cols_train
                                X_test_subset_for_scaling = X_test[cols_to_scale_in_test]
                                X_test_scaled_values = scaler.transform(X_test_subset_for_scaling)
                                X_test[cols_to_scale_in_test] = X_test_scaled_values
                            # Handle missing/extra columns if necessary, for now assume they match or subset

                st.session_state.update({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test})

                # Define models based on problem type
                if st.session_state.problem_type == "Classification":
                    models_to_train = {
                        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
                        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                        "Random Forest": RandomForestClassifier(random_state=random_state),
                        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
                        "Support Vector Machine": SVC(random_state=random_state, probability=True),
                        "K-Nearest Neighbors": KNeighborsClassifier(),
                        "Gaussian Naive Bayes": GaussianNB()
                    }
                    scoring = 'accuracy'
                else: # Regression
                    # Local imports for LinearRegression, Ridge, RandomForestRegressor, etc.
                    # are removed as these models are now imported globally by the first search/replace block.
                    # ElasticNet is also imported globally.
                    models_to_train = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(random_state=random_state),
                        "ElasticNet Regression": ElasticNet(random_state=random_state),
                        "Random Forest Regressor": RandomForestRegressor(random_state=random_state),
                        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=random_state),
                        "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
                        "Support Vector Regressor": SVR(),
                        "K-Nearest Neighbors Regressor": KNeighborsRegressor()
                    }
                    scoring = 'r2'

                trained_models = {}
                model_scores_dict = {}
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (name, model) in enumerate(models_to_train.items()):
                    status_text.text(f"Training {name}...")
                    model.fit(X_train, y_train)
                    trained_models[name] = model
                    
                    y_pred_test = model.predict(X_test)
                    y_proba_test = model.predict_proba(X_test) if hasattr(model, 'predict_proba') and st.session_state.problem_type == "Classification" else None
                    
                    metrics = get_model_metrics(y_test, y_pred_test, y_proba_test, problem_type=st.session_state.problem_type)
                    cv_score = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring).mean()
                    
                    current_model_scores = {'CV Mean Score': cv_score}
                    current_model_scores.update(metrics) # Add all relevant metrics
                    model_scores_dict[name] = current_model_scores

                    progress_bar.progress((i + 1) / len(models_to_train))
                
                st.session_state.models = trained_models
                st.session_state.model_scores = model_scores_dict

                # Determine best model
                if st.session_state.problem_type == "Classification":
                    best_model_name = max(model_scores_dict, key=lambda k: (model_scores_dict[k]['Test Accuracy'] or 0, model_scores_dict[k]['Test AUC'] or 0))
                else: # Regression
                    # Ensure 'R2' exists and provide a default if not (e.g., for models where R2 might not be applicable or calculable)
                    best_model_name = max(model_scores_dict, key=lambda k: model_scores_dict[k].get('R2', -float('inf')))
                
                st.session_state.best_model_info = {
                    'name': best_model_name,
                    'model': trained_models[best_model_name],
                    'metrics': model_scores_dict[best_model_name]
                }
                status_text.text("Training completed!")
                st.success(f"✅ Training completed! Best model: {best_model_name}")

            except Exception as e:
                st.error(f"Error during training: {e}")
                import traceback
                st.error(traceback.format_exc())

def model_comparison_page():
    st.header("📊 Model Comparison")
    if not st.session_state.model_scores:
        st.warning("⚠️ Please train models first.")
        return

    scores_df = pd.DataFrame(st.session_state.model_scores).T.fillna(0) # Fill NaN with 0 for display
    scores_df = scores_df.round(4)

    st.subheader("🏆 Model Leaderboard")
    if st.session_state.problem_type == "Classification":
        sort_by = 'Test Accuracy'
        display_cols = ['CV Mean Score', 'Test Accuracy', 'Test F1-score', 'Test AUC']
    else: # Regression
        sort_by = 'R2'
        display_cols = ['CV Mean Score', 'R2', 'MSE'] # Add other relevant regression metrics if needed
        # Ensure MSE is present, if not, it will be filled with 0 by .fillna(0) earlier or handle missing more gracefully if needed
    
    leaderboard = scores_df[display_cols].sort_values(by=sort_by, ascending=False)
    leaderboard['Rank'] = range(1, len(leaderboard) + 1)
    leaderboard = leaderboard[['Rank'] + display_cols]
    st.dataframe(leaderboard.style.background_gradient(subset=[sort_by], cmap='RdYlGn'), use_container_width=True)

    best_model_name = st.session_state.best_model_info['name']
    best_metric_val = st.session_state.best_model_info['metrics'].get(sort_by, 'N/A')
    st.markdown(f"<div class='success-message'><h4>🥇 Best Model: {best_model_name} ({sort_by}: {best_metric_val:.4f})</h4></div>", unsafe_allow_html=True)

    st.subheader("📈 Performance Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = scores_df[sort_by].sort_values(ascending=True)
    bars = ax.barh(plot_data.index, plot_data.values, color=['#ff6b6b' if idx == best_model_name else '#4ecdc4' for idx in plot_data.index])
    ax.set_xlabel(sort_by)
    ax.set_title('Model Performance Comparison')
    st.pyplot(fig)

    if st.session_state.problem_type == "Classification" and st.session_state.X_test is not None:
        st.subheader(f"📋 Detailed Metrics for Best Model: {best_model_name}")
        best_model = st.session_state.best_model_info['model']
        y_pred = best_model.predict(st.session_state.X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.text("Classification Report:")
            report_df = pd.DataFrame(classification_report(st.session_state.y_test, y_pred, output_dict=True)).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
        with col2:
            st.text("Confusion Matrix:")
            cm = confusion_matrix(st.session_state.y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)

def explainability_page():
    st.header("🔍 Model Explainability (SHAP)")
    if not st.session_state.best_model_info or st.session_state.X_test is None:
        st.warning("⚠️ Please train a model and ensure test data is available.")
        return

    best_model = st.session_state.best_model_info['model']
    best_model_name = st.session_state.best_model_info['name']
    X_test_df = pd.DataFrame(st.session_state.X_test, columns=st.session_state.X_train.columns)

    st.write(f"**Explaining model:** {best_model_name}")
    with st.spinner("Generating SHAP explanations..."):
        try:
            # SHAP Explainer
            if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier,
                                      RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor)):
                explainer = shap.TreeExplainer(best_model)
            elif isinstance(best_model, (LogisticRegression, LinearRegression, Ridge, ElasticNet)):
                explainer = shap.LinearExplainer(best_model, X_test_df) # Pass data for LinearExplainer
            elif isinstance(best_model, (SVC, SVR, KNeighborsClassifier, KNeighborsRegressor, GaussianNB)):
                 # KernelExplainer can be slow or not directly applicable for some, use a subset of X_train for background data
                 # For KNN and Naive Bayes, KernelExplainer is a common choice for SHAP if TreeExplainer/LinearExplainer aren't suitable.
                background_data = shap.sample(st.session_state.X_train, min(100, len(st.session_state.X_train)))
                if isinstance(background_data, np.ndarray):
                    background_data = pd.DataFrame(background_data, columns=X_test_df.columns)
                explainer = shap.KernelExplainer(best_model.predict_proba if hasattr(best_model, 'predict_proba') else best_model.predict, background_data)
            else:
                st.error(f"SHAP explanations not supported for {best_model_name} with current setup.")
                return

            shap_values = explainer.shap_values(X_test_df)
            
            # For binary classification, shap_values might be a list of two arrays (for class 0 and 1)
            # We typically use shap_values for the positive class (class 1)
            if isinstance(shap_values, list) and len(shap_values) == 2 and st.session_state.problem_type == "Classification":
                shap_values_plot = shap_values[1]
            else:
                shap_values_plot = shap_values

            st.subheader("📊 Global Feature Importance (SHAP Summary Plot)")
            fig_summary, ax_summary = plt.subplots()
            shap.summary_plot(shap_values_plot, X_test_df, plot_type="bar", show=False, max_display=15)
            st.pyplot(fig_summary)

            st.subheader("🎯 SHAP Beeswarm Plot")
            fig_beeswarm, ax_beeswarm = plt.subplots()
            shap.summary_plot(shap_values_plot, X_test_df, show=False, max_display=15)
            st.pyplot(fig_beeswarm)

            st.subheader("💧 Individual Prediction Explanation (Waterfall Plot)")
            sample_idx = st.selectbox("Select a sample from test set to explain:", range(min(20, len(X_test_df))))
            if st.button("Explain Sample"):
                fig_waterfall, ax_waterfall = plt.subplots()
                # Create SHAP Explanation object
                if isinstance(explainer, shap.explainers.Tree):
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, list): # Multi-output case for TreeExplainer
                        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                elif isinstance(explainer, shap.explainers.Linear) or isinstance(explainer, shap.explainers.Kernel):
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, np.ndarray) and expected_value.ndim > 0:
                         expected_value = expected_value[0] # Take the first if it's an array
                else:
                    expected_value = 0 # Fallback, might need adjustment
                
                shap_explanation_obj = shap.Explanation(
                    values=shap_values_plot[sample_idx],
                    base_values=expected_value,
                    data=X_test_df.iloc[sample_idx].values,
                    feature_names=X_test_df.columns
                )
                shap.waterfall_plot(shap_explanation_obj, show=False, max_display=15)
                st.pyplot(fig_waterfall)

                actual = st.session_state.y_test.iloc[sample_idx]
                predicted = best_model.predict(X_test_df.iloc[[sample_idx]])[0]
                st.metric("Actual Value", f"{actual:.2f}")
                st.metric("Predicted Value", f"{predicted:.2f}")

        except Exception as e:
            st.error(f"Error generating SHAP explanations: {e}")
            import traceback
            st.error(traceback.format_exc())

def model_export_page():
    st.header("💾 Model Export")
    if not st.session_state.best_model_info:
        st.warning("⚠️ Please train a model first.")
        return

    best_model_info = st.session_state.best_model_info
    best_model = best_model_info['model']
    best_model_name = best_model_info['name']

    st.write(f"**Best Model:** {best_model_name}")
    st.write(f"**Metrics:**")
    st.json(best_model_info['metrics'])

    # Build a pipeline for export (model + scaler if used)
    from sklearn.pipeline import Pipeline
    steps = []
    if st.session_state.scaler:
        steps.append(('scaler', st.session_state.scaler))
    steps.append(('model', best_model))
    pipeline_to_export = Pipeline(steps)
    st.session_state.trained_pipeline = pipeline_to_export

    export_format = st.selectbox("Choose export format:", ["Joblib (.joblib)", "Pickle (.pkl)"])
    file_name_suggestion = f"{best_model_name.lower().replace(' ', '_')}_pipeline"
    file_name = st.text_input("Enter filename for export:", value=file_name_suggestion)

    if st.button("📥 Download Model Pipeline", type="primary"):
        try:
            buffer = io.BytesIO()
            ext = ".joblib" if "Joblib" in export_format else ".pkl"
            if ext == ".joblib":
                joblib.dump(pipeline_to_export, buffer)
            else:
                import pickle
                pickle.dump(pipeline_to_export, buffer)
            
            buffer.seek(0)
            st.download_button(
                label=f"Download {file_name}{ext}",
                data=buffer,
                file_name=f"{file_name}{ext}",
                mime="application/octet-stream"
            )
            st.success("Model pipeline ready for download!")
        except Exception as e:
            st.error(f"Error exporting model: {e}")

    st.subheader("📖 How to use the exported pipeline:")
    st.code(f"""
import joblib # or import pickle
import pandas as pd

# Load the pipeline
pipeline = joblib.load('{file_name}{'.joblib' if 'Joblib' in export_format else '.pkl'}')

# Example new data (must have same columns as training, BEFORE scaling)
# new_data = pd.DataFrame(...) 

# Preprocess new_data similar to training (handle categoricals, ensure column order)
# Ensure new_data has columns: {list(st.session_state.X_train.columns) if st.session_state.X_train is not None else 'X_train_columns'}

# Make predictions
# predictions = pipeline.predict(new_data)
# print(predictions)
""", language='python')

# --- Main Application ---
def main():
    init_session_state()
    st.markdown('<h1 class="main-header">🤖 AutoML & Explainability Platform</h1>', unsafe_allow_html=True)

    st.sidebar.title("⚙️ Workflow")
    page_options = ["Data Upload & Preview", "Model Training", "Model Comparison", "Explainability", "Model Export"]
    
    # Handle auto-run navigation
    if st.session_state.get('auto_run_triggered') and st.session_state.target_column:
        st.session_state.auto_run_triggered = False # Reset trigger
        st.session_state.current_page = "Model Training"
        st.session_state.auto_run_triggered_for_training = True # Signal model_training_page to auto-start
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Data Upload & Preview"

    page = st.sidebar.radio("Navigate", page_options, key='navigation_radio', index=page_options.index(st.session_state.current_page))
    st.session_state.current_page = page # Update current page based on user selection

    if page == "Data Upload & Preview":
        data_upload_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Model Comparison":
        model_comparison_page()
    elif page == "Explainability":
        explainability_page()
    elif page == "Model Export":
        model_export_page()

    st.sidebar.markdown("---_Developed with Trae AI_---")

if __name__ == "__main__":
    main()