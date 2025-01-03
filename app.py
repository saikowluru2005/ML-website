# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import numpy as np
# from sklearn.neighbors import KNeighborsRegressor

# st.title('Linear Regression Web App')
# option=st.selectbox("Select model",["Liner regression", "KNN"])
# st.title(option)
# uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
# if option=="Liner regression":
#     if uploaded_file:
        
#         df = pd.read_csv(uploaded_file)
#         st.write("### Data Preview")
#         st.write(df.head())

#         target_variable = st.selectbox("Select the target variable (Y)", df.columns)
        
#         feature_variables = st.multiselect("Select the feature variables (X)", df.columns, default=[col for col in df.columns if col != target_variable])
        
#         if target_variable and len(feature_variables) > 0:
#             X = df[feature_variables]
#             y = df[target_variable]


#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            

#             model = LinearRegression()
#             model.fit(X_train, y_train)
    
#             predictions = model.predict(X_test)
#             mse = mean_squared_error(y_test, predictions)

#             st.write("### Model Performance")
#             st.write(f"Mean Squared Error: {mse:.2f}")


#             st.write("### Make Predictions")
#             input_data = {}
#             for feature in feature_variables:
#                 value = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))
#                 input_data[feature] = value
#             input_df = pd.DataFrame([input_data])
#             prediction_result = model.predict(input_df)[0]

#             st.write("### Prediction Result")
#             st.write(f"Predicted {target_variable}: {prediction_result:.2f}")
# if option=="KNN":
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         st.write("### Data Preview")
#         st.write(df.head())
#         target_variable = st.selectbox("Select the target variable (Y)", df.columns)
#         feature_variables = st.multiselect("Select the feature variables (X)", df.columns, default=[
#             col for col in df.columns if col != target_variable])
#         if target_variable and len(feature_variables) > 0:
#             X = df[feature_variables]
#             y = df[target_variable]
#             X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                                 test_size=0.2, random_state=42)
#             model = KNeighborsRegressor()
#             model.fit(X_train, y_train)
#             predictions = model.predict(X_test)
#             mse = mean_squared_error(y_test, predictions)
#             st.write("### Model Performance")
#             st.write(f"Mean Squared Error: {mse:.2f}")
#             st.write("### Make Predictions")
#             input_data = {}
#             for feature in feature_variables:
#                 value = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))
#                 input_data[feature] = value
#                 input_df = pd.DataFrame([input_data])
#                 prediction_result = model.predict(input_df)[0]
#                 st.write("### Prediction Result")
#                 st.write(f"Predicted {target_variable}: {prediction_result:.2f}")
            
    
                    
# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import LabelEncoder
# import numpy as np

# st.title('Linear Regression Web App')
# option = st.selectbox("Select model", ["Linear regression", "KNN"])
# st.title(option)
# uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.write("### Data Preview")
#     st.write(df.head())

#     target_variable = st.selectbox("Select the target variable (Y)", df.columns)
#     feature_variables = st.multiselect("Select the feature variables (X)", df.columns, default=[col for col in df.columns if col != target_variable])

#     if target_variable and len(feature_variables) > 0:
#         X = df[feature_variables]
#         y = df[target_variable]

        
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col])

#         if y.dtype == 'object':
#             le_y = LabelEncoder()
#             y = le_y.fit_transform(y)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         if option == "Linear regression":
#             model = LinearRegression()
#         elif option == "KNN":
#             model = KNeighborsRegressor()

#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         mse = mean_squared_error(y_test, predictions)

#         st.write("### Model Performance")
#         st.write(f"Mean Squared Error: {mse:.2f}")

#         st.write("### Make Predictions")
#         input_data = {}
#         for feature in feature_variables:
#             value = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))
#             input_data[feature] = value
        
#         input_df = pd.DataFrame([input_data])

#         for col in input_df.columns:
#             if df[col].dtype == 'object':
#                 le = LabelEncoder()
#                 input_df[col] = le.fit_transform(input_df[col])

#         prediction_result = model.predict(input_df)[0]

#         st.write("### Prediction Result")
#         st.write(f"Predicted {target_variable}: {prediction_result:.2f}")



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.title('Regression Web App')
option = st.selectbox("Select model", ["Linear regression", "KNN"])
st.title(option)
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

    st.write("### Data Visualizations")
    st.write("#### Pairplot")
    sns.pairplot(df)
    st.pyplot(plt)

    st.write("#### Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

    target_variable = st.selectbox("Select the target variable (Y)", df.columns)
    feature_variables = st.multiselect("Select the feature variables (X)", df.columns, default=[col for col in df.columns if col != target_variable])

    if target_variable and len(feature_variables) > 0:
        X = df[feature_variables]
        y = df[target_variable]

        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        if y.dtype == 'object':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if option == "Linear regression":
            model = LinearRegression()
        elif option == "KNN":
            model = KNeighborsRegressor()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        st.write("### Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")

        st.write("#### Predictions vs Actual Values")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions)
        plt.plot(y_test, y_test, color='red')  # Identity line
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predictions vs Actual Values")
        st.pyplot(plt)

        st.write("### Make Predictions")
        input_data = {}
        for feature in feature_variables:
            value = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))
            input_data[feature] = value
        
        input_df = pd.DataFrame([input_data])

        for col in input_df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                input_df[col] = le.fit_transform(input_df[col])

        prediction_result = model.predict(input_df)[0]

        st.write("### Prediction Result")
        st.write(f"Predicted {target_variable}: {prediction_result:.2f}")
