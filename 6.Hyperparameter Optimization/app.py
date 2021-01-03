import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_option("deprecation.showPyplotGlobalUse", False)

from sklearn.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    precision_score,
    recall_score,
)


def main():
    st.title("Hyperparameter Optimization Web App")
    st.sidebar.title("Algorithm and Hyperparameters Selections")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("mushrooms_clas.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.iloc[:, 0]
        x = df.iloc[:, 1:]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0
        )
        return x_train, x_test, y_train, y_test

    @st.cache(persist=True)
    def split_reg(df):
        y = df.iloc[:, 0]
        x = df.iloc[:, 1:]
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        x = scaler_x.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0
        )
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

        if "Scatter Plot" in metrics_list:
            st.subheader("Visualisation of Actual and Predicted Values")
            plt.scatter(y_test, y_pred)
            plt.xlabel("Y Test")
            plt.ylabel("Predicted Y")
            st.pyplot()

        if "Line Plot" in metrics_list:
            st.subheader("Visualisation of Actual and Predicted Values")
            plt.plot(y_test, label="Test")
            plt.plot(y_pred.reshape(-1, 1), label="Predict")
            # plt.xlabel('Y Test')
            # plt.ylabel('Predicted Y')
            st.pyplot()

    st.sidebar.subheader("Classifiction or Regression")
    Clas_Reg = st.sidebar.selectbox("", ("Classifiction", "Regression"))

    if Clas_Reg == "Regression":
        st.markdown("## Energy Output Prediction of Plants")

        df = pd.read_csv("pow_plant_reg.csv")
        x_train, x_test, y_train, y_test = split_reg(df)

        st.sidebar.subheader("Choose Regression")
        regression = st.sidebar.selectbox(
            "",
            (
                "Linear Regression",
                "Support Vektor",
                "Decision Tree",
                "Random Forest",
            ),
        )

        if regression == "Decision Tree":
            st.sidebar.subheader("Model Hyperparameters")
            splitter = st.sidebar.radio(
                "Splitter(default=best)", ("best", "random"), key="splitter"
            )
            criterion = st.sidebar.radio(
                "Criterion(default=mse)",
                ("friedman_mse", "mse", "mae"),
                key="criterion",
            )

            metrics = st.sidebar.multiselect(
                "Visualization",
                ("Line Plot", "Scatter Plot"),
            )

            if st.sidebar.button("Run", key="run"):
                st.subheader("Decision Tree Results")
                model = DecisionTreeRegressor(splitter=splitter, criterion=criterion)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                st.write("MAE : ", mean_absolute_error(y_test, y_pred).round(2))
                st.write(
                    "RMSE : ",
                    np.sqrt(mean_squared_error(y_test, y_pred)).round(2),
                )
                st.write(
                    "R2_Score : ",
                    r2_score(y_test, y_pred).round(2),
                )
                plot_metrics(metrics)

        if st.sidebar.checkbox("Show Raw Data", False):
            st.subheader("Power Plant Data Set")
            st.write(df.head(), df.shape, "null values:", df.isnull().sum().sum())

    if Clas_Reg == "Classifiction":
        st.markdown("## Mushroom Classification")

        df = load_data()
        x_train, x_test, y_train, y_test = split(df)
        class_names = ["edible", "poisonous"]

        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox(
            "Classifier", ("K-NN", "SVM", "Logistic Regression", "Random Forest")
        )

        if classifier == "K-NN":
            st.sidebar.subheader("Model Hyperparameters")
            n_neighbors = st.sidebar.number_input(
                "Number of neighbors(Default=5)", 2, 10, step=1, key="n_neighbors"
            )

            metrics = st.sidebar.multiselect(
                "What metrics to plot?",
                ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
            )

            if st.sidebar.button("Classify", key="classify"):
                st.subheader("K-NN Results")
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy : ", accuracy.round(2))
                st.write(
                    "Precision : ",
                    precision_score(y_test, y_pred, labels=class_names).round(2),
                )
                st.write(
                    "Recall : ",
                    recall_score(y_test, y_pred, labels=class_names).round(2),
                )
                plot_metrics(metrics)

        if classifier == "SVM":
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input(
                "C (Regularization Parameter)", 0.01, 10.0, step=0.01, key="C"
            )
            kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
            gamma = st.sidebar.radio(
                "Gamma(Kernel Coefficient)", ("scale", "auto"), key="gamma"
            )

            metrics = st.sidebar.multiselect(
                "What metrics to plot?",
                ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
            )

            if st.sidebar.button("Classify", key="classify"):
                st.subheader("SVM Results")
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy : ", accuracy.round(2))
                st.write(
                    "Precision : ",
                    precision_score(y_test, y_pred, labels=class_names).round(2),
                )
                st.write(
                    "Recall : ",
                    recall_score(y_test, y_pred, labels=class_names).round(2),
                )
                plot_metrics(metrics)

        if classifier == "Logistic Regression":
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input(
                "C (Regularization Parameter)", 0.01, 10.0, step=0.01, key="C_RL"
            )
            max_iter = st.sidebar.slider(
                "Maximum number of iteration", 100, 500, key="max_iter"
            )

            metrics = st.sidebar.multiselect(
                "What metrics to plot?",
                ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
            )

            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(C=C, max_iter=max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy : ", accuracy.round(2))
                st.write(
                    "Precision : ",
                    precision_score(y_test, y_pred, labels=class_names).round(2),
                )
                st.write(
                    "Recall : ",
                    recall_score(y_test, y_pred, labels=class_names).round(2),
                )
                plot_metrics(metrics)

        if classifier == "Random Forest":
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.number_input(
                "The number of trees", 100, 5000, step=10, key="n_estimators"
            )
            max_depth = st.sidebar.number_input(
                "The maximum depth of tree", 1, 20, step=1, key="max_depth"
            )
            bootstrap = st.sidebar.radio(
                "Bootstrap samples when building trees",
                ("True", "False"),
                key="bootstrap",
            )

            metrics = st.sidebar.multiselect(
                "What metrics to plot?",
                ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
            )

            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    bootstrap=bootstrap,
                    n_jobs=-1,
                )
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy : ", accuracy.round(2))
                st.write(
                    "Precision : ",
                    precision_score(y_test, y_pred, labels=class_names).round(2),
                )
                st.write(
                    "Recall : ",
                    recall_score(y_test, y_pred, labels=class_names).round(2),
                )
                plot_metrics(metrics)

        if st.sidebar.checkbox("Show Raw Data", False):
            st.subheader("Mushrooms Data Set")
            st.write(df.head(), df.shape, "null values:", df.isnull().sum().sum())


if __name__ == "__main__":
    main()
