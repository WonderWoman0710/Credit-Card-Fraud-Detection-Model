import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Function to load data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Load data
# Plotting
st.title('Credit Card Fraud Detection')

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Data preprocessing
    data.drop(columns=['Time'], inplace=True)
    data.drop_duplicates(inplace=True)

    # Undersampling
    legal = data[data['Class'] == 0]
    fraud = data[data['Class'] == 1]
    legal_sample = legal.sample(n=473)
    data_undersample = pd.concat([legal_sample, fraud], ignore_index=True)

    # Splitting data
    x = data_undersample.drop(columns=['Class'])
    y = data_undersample['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Scaling data
    scaler = StandardScaler()
    x_train_scale = scaler.fit_transform(x_train)
    x_test_scale = scaler.fit_transform(x_test)

    # Model training
    model = LogisticRegression()
    model.fit(x_train_scale, y_train)
    pred = model.predict(x_test_scale)

    # Accuracy
    accuracy = accuracy_score(pred, y_test)

    # Class Distribution
    st.subheader('Class Distribution Before Balancing')
    st.bar_chart(data['Class'].value_counts())

    st.subheader('Class Distribution After Undersampling')
    st.bar_chart(data_undersample['Class'].value_counts())

    # Amount Distribution
    st.subheader('Amount Distribution for Legal Transactions')
    legal_amount_plot = sns.boxplot(x='Class', y='Amount', data=legal)
    st.pyplot(legal_amount_plot.figure)

    st.subheader('Amount Distribution for Fraud Transactions')
    fraud_amount_plot = sns.boxplot(x='Class', y='Amount', data=fraud)
    st.pyplot(fraud_amount_plot.figure)

    # Confusion Matrix
    st.subheader('Confusion Matrix of Logistic Regression Model')
    st.write('Actual vs. Predicted')
    st.write(pd.DataFrame({'Actual': y_test, 'Predicted': pred}))

    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    # Display confusion matrix values
    st.write(f'True Negatives: {tn}')
    st.write(f'False Positives: {fp}')
    st.write(f'False Negatives: {fn}')
    st.write(f'True Positives: {tp}')

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', xticklabels=['Legal', 'Fraud'], yticklabels=['Legal', 'Fraud'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
