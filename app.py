import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the pre-trained models
rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
xgb_model = pickle.load(open('xgboost_model.pkl', 'rb'))
logistic_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
dt_model = pickle.load(open('decision_tree_model.pkl', 'rb'))

# Load the LabelEncoder and StandardScaler
le = pickle.load(open('label_encoder.pkl', 'rb'))
scaler = pickle.load(open('standard_scaler.pkl', 'rb'))

@st.cache(allow_output_mutation=True)
def load_data():
    # Load your dataset here
    df = pd.read_csv('C:/Users/Aditya Rai/Dropbox/PC/Desktop/Bank-Term-Deposit-Prediction/dataset/bank.csv')
    return df

def build_model(train_size, random_state):
    df = load_data()

    # Feature Engineering and Preprocessing
    df.drop(['default'], axis=1, inplace=True)
    df.drop(['pdays'], axis=1, inplace=True)
    df = df[df['campaign'] < 33]
    df = df[df['previous'] < 31]

    # Label Encoding
    le = LabelEncoder()
    df['job'] = le.fit_transform(df['job'])
    df['marital'] = le.fit_transform(df['marital'])
    df['education'] = le.fit_transform(df['education'])
    df['contact'] = le.fit_transform(df['contact'])
    df['month'] = le.fit_transform(df['month'])
    df['poutcome'] = le.fit_transform(df['poutcome'])
    df['housing'] = le.fit_transform(df['housing'])
    df['loan'] = le.fit_transform(df['loan'])
    df['deposit'] = le.fit_transform(df['deposit'])

    # Standardization
    sc = StandardScaler()
    df['balance'] = sc.fit_transform(df['balance'].values.reshape(-1, 1))
    df['duration'] = sc.fit_transform(df['duration'].values.reshape(-1, 1))

    # Splitting Data Into Train and Test
    X = df.drop(['deposit'], axis=1)
    y = df['deposit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# Streamlit app
def main():
    st.title("Bank Marketing Term Deposit Prediction App")

    # Input form
    train_size = st.slider("Train Size", 0.1, 0.9, 0.8, step=0.1)
    random_state = st.slider("Random State", 0, 100, 0)
    
    X_train, X_test, y_train, y_test = build_model(train_size=train_size, random_state=random_state)

    # Rest of your Streamlit app code...

if __name__ == "__main__":
    main()
