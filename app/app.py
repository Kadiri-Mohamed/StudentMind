from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="StudentMind",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student_mental_health.csv"
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.joblib"

TARGET = "Mental_Health_Risk"
ID_COLUMN = "Student_ID"


# =====================================================
# 2. CHARGEMENT DES DONNÉES ET DU MODÈLE
# =====================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()

except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()


# =====================================================
# 3. PRÉPARATION DES DONNÉES POUR L'ÉVALUATION
# =====================================================

if TARGET not in df.columns:
    st.error(f"La colonne cible '{TARGET}' est introuvable.")
    st.stop()

df = df.dropna(subset=[TARGET]).drop_duplicates()

X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)


# =====================================================
# 4. SIDEBAR / NAVIGATION
# =====================================================

st.sidebar.title("StudentMind")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction"]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "StudentMind - Projet de classification supervisée"
)


# =====================================================
# 5. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("StudentMind Dashboard")

    st.write(
        "Visualisation des performances du modèle "
        "de classification."
    )

    st.markdown("---")

    # KPI

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision Macro",
        f"{report['macro avg']['precision']:.2f}"
    )

    col3.metric(
        "Recall Macro",
        f"{report['macro avg']['recall']:.2f}"
    )

    col4.metric(
        "F1 Macro",
        f"{report['macro avg']['f1-score']:.2f}"
    )

    st.markdown("---")

    # Distribution des classes

    st.subheader("Distribution des classes")

    class_counts = y.value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(class_counts)

    with col2:
        st.dataframe(
            class_counts.rename("Nombre").to_frame(),
            use_container_width=True
        )

    st.markdown("---")

    # Rapport de classification

    st.subheader("Classification Report")

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.markdown("---")

    # Matrice de confusion

    st.subheader("Matrice de confusion")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Valeurs réelles")
    ax.set_title("Matrice de confusion")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.info(
        "Les résultats présentés correspondent à une évaluation "
        "sur le jeu de test du dataset."
    )


# =====================================================
# 6. PREDICTION
# =====================================================

elif page == "Prediction":

    st.title("Tester le modèle")

    st.write(
        "Saisissez les caractéristiques d'une observation "
        "pour tester le fonctionnement du modèle."
    )

    st.warning(
        "Cette démonstration utilise un dataset synthétique. "
        "Les prédictions ne constituent ni un diagnostic médical, "
        "ni un dépistage clinique, ni une évaluation individuelle."
    )

    st.markdown("---")

    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    input_data = {}

    with st.form("prediction_form"):

        st.subheader("Informations générales")

        col1, col2 = st.columns(2)

        for index, feature in enumerate(categorical_features):

            options = sorted(
                X[feature].dropna().unique().tolist(),
                key=str
            )

            if not options:
                continue

            with (col1 if index % 2 == 0 else col2):
                input_data[feature] = st.selectbox(
                    feature.replace("_", " "),
                    options
                )

        st.markdown("---")

        st.subheader("Informations numériques")

        col1, col2 = st.columns(2)

        for index, feature in enumerate(numeric_features):

            values = pd.to_numeric(
                X[feature],
                errors="coerce"
            ).dropna()

            if values.empty:
                continue

            minimum = float(values.min())
            maximum = float(values.max())
            default = float(values.median())

            is_integer = pd.api.types.is_integer_dtype(
                X[feature]
            )

            with (col1 if index % 2 == 0 else col2):

                input_data[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=1.0 if is_integer else 0.1
                )

        st.markdown("---")

        submitted = st.form_submit_button(
            "Lancer la prédiction",
            use_container_width=True
        )

    # Résultat

    if submitted:

        input_df = pd.DataFrame(
            [input_data],
            columns=X.columns
        )

        try:

            prediction = model.predict(input_df)[0]

            st.subheader("Résultat de la démonstration")

            st.success(
                f"Classe prédite : {prediction}"
            )

            # Probabilités si disponibles

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_df
                )[0]

                probability_df = pd.DataFrame({
                    "Classe": model.classes_,
                    "Probabilité": probabilities
                })

                st.subheader("Probabilités prédites")

                st.bar_chart(
                    probability_df.set_index("Classe")
                )

                st.dataframe(
                    probability_df.style.format({
                        "Probabilité": "{:.2%}"
                    }),
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Impossible de réaliser la prédiction : {e}"
            )