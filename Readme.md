# StudentMind

### Student Mental Health Risk Prediction

StudentMind is a Machine Learning project that analyzes student-related factors and predicts mental health risk levels using supervised classification algorithms.

The project explores the relationships between academic performance, lifestyle habits, social behavior, and psychosocial factors to estimate student mental health risk categories.

The classification model predicts one of three risk levels:

- **Low**
- **Medium**
- **High**

---

## Project Objectives

- Analyze student data and identify relevant patterns.
- Explore factors associated with mental health risk levels.
- Perform data cleaning and preprocessing.
- Train and compare supervised classification models.
- Optimize model performance using hyperparameter tuning.
- Evaluate model reliability using classification metrics.
- Automate the Machine Learning workflow with Apache Airflow.
- Provide an interactive interface using Streamlit.

---

## Dataset

**Source:** [Mental Health Risk Prediction Dataset (Students)](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)

**File:** `student_mental_health.csv`

| Property | Value |
|---|---|
| Number of records | 2,000 |
| Number of columns | 17 |
| Input features | 16 |
| Target variable | Mental_Health_Risk |
| Problem type | Multiclass Classification |
| Target classes | Low, Medium, High |
| Data format | CSV |

### Features

| Category | Variables |
|---|---|
| Demographic | Age, Gender |
| Academic | Year_of_Study, GPA, Study_Hours_Per_Week, Attendance_Rate |
| Lifestyle | Sleep_Hours, Social_Media_Hours, Screen_Time_Hours, Physical_Activity, Diet_Quality |
| Psychosocial | Financial_Stress, Family_Support, Part_Time_Job, Previous_Mental_Health_History |
| Target | Mental_Health_Risk |

`Student_ID` is used only as an identifier and is excluded from model training.

---

## Project Workflow

The project follows a complete Machine Learning workflow.

### 1. Data Understanding & EDA
- Dataset exploration and statistical analysis.
- Missing values and duplicate detection.
- Outlier analysis.
- Target distribution analysis.
- Feature relationships and correlation analysis.

### 2. Data Preprocessing
- Missing value imputation.
- Categorical feature encoding.
- Numerical feature scaling.
- Outlier treatment when justified.
- Class distribution analysis.
- Train/Test split (80% / 20%).
- Preprocessing pipeline construction.

### 3. Model Training
Training and comparison of multiple classification algorithms:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

### 4. Model Optimization
- Cross-validation.
- GridSearchCV / RandomizedSearchCV.
- Hyperparameter optimization.
- Model selection.

### 5. Model Evaluation
Models are evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix
- ROC-AUC (One-vs-Rest), when appropriate.

### 6. Model Serialization
The complete preprocessing and classification pipeline is saved using Joblib for reuse in the Streamlit application.

### 7. Workflow Automation
Apache Airflow orchestrates the main Machine Learning stages, including data preparation, training, and evaluation.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine Learning |
| XGBoost | Gradient Boosting |
| Imbalanced-learn | Class imbalance handling |
| Joblib | Model serialization |
| Streamlit | Interactive application |
| Apache Airflow | Workflow orchestration |
| Docker | Containerization |
| Git & GitHub | Version control |

---

## Project Structure

```text
StudentMind/
│
├── Data/
│   └── student_mental_health.csv
│
├── Notebooks/
│   └── StudentMind.ipynb
│
├── Models/
│   └── best_pipeline.joblib
│
├── dags/
│   └── studentmind_dag.py
│
├── app.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
│
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Docker and Docker Compose (optional)

### 1. Clone the repository

```bash
git clone <repository-url>
cd StudentMind
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the CSV file from Kaggle and place it inside:

```text
Data/student_mental_health.csv
```

### 5. Launch the Streamlit application

```bash
streamlit run app.py
```

### 6. Run with Docker

```bash
docker compose up --build
```

---

## Airflow Workflow

Apache Airflow is used to automate and monitor the Machine Learning pipeline.

The workflow includes:

```text
Data Validation
      |
      v
Data Preparation
      |
      v
Preprocessing
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
Model Saving
```

Airflow provides task scheduling, execution monitoring, logs, and error tracking.

---

## Streamlit Application

StudentMind provides an interactive dashboard with the following features:

**Dashboard**
- Dataset overview.
- Risk level distribution.
- Data visualization.

**Model Evaluation**
- Classification metrics.
- Model comparison.
- Confusion matrix.
- Performance charts.

**Prediction**
- Student feature input form.
- Input validation.
- Predicted risk category.
- Prediction probabilities.

---

## Reproducibility & Quality

The project follows several practices to ensure reliable and reproducible experiments:

- Fixed random states.
- Stratified train/test splitting.
- Cross-validation.
- Scikit-learn preprocessing pipelines.
- Protection against data leakage.
- Complete pipeline serialization.
- Dependency management.

---

## Limitations & Responsible Use

The dataset consists of synthetically generated student profiles. Consequently, the model learns patterns from simulated data that may not represent real-world student populations.

StudentMind is not a clinical diagnostic or screening system.

Its predictions must not be used for medical decisions, student selection, disciplinary actions, or individual profiling.

The results should be interpreted as statistical outputs rather than clinical assessments.

---

## Project Timeline

| Activity | Date |
|---|---|
| Data Understanding & EDA | 09–10 November 2026 |
| Data Preprocessing | 10–11 November 2026 |
| Model Training | 11–12 November 2026 |
| Optimization & Evaluation | 12–13 November 2026 |
| Airflow & Streamlit | 13–14 November 2026 |
| Documentation & Finalization | 14 November 2026 |
| Project Presentation | 16 November 2026 |

---

## Author

**KADIRI Mohamed**

Machine Learning & Data Science

---

## Dataset Reference

Maqdad Ali — [Mental Health Risk Prediction Dataset (Students)](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)