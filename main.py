import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from io import BytesIO

# Custom CSS styles
st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: grey;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2F5496;
    }
    .header {
        color: #2F5496;
    }
    .text {
        font-family: Arial, sans-serif;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


header = st.container()

#@st.cache_data
#def get_data():
#    df_crime_num_2 = pd.read_csv("data/crime_data.csv")
#    return df_crime_num_2

with header:
    st.title('The NZ Police Victimisation Dataset')
    st.write("""
    blurb
    """)

dataset = st.expander('The Data', expanded=False)
with dataset:
    st.write("""The Data Frame:""")
    #df_crime_num_2  = get_data()
    df_crime_num_2 = pd.read_csv("data/crime_data.csv")
    st.write(df_crime_num_2.head(5))

features = st.expander('Data Sources', expanded=False)
with features:
    #st.header('Features')
    st.write("""
    This dataframe was used by merging several datasets. Police crime data, sourced from the NZ Police
    website, Police numbers sourced from the NZ Police Annual Report and, Household Living Costs and the Unemployment
    Rate are sourced from NZ Statistics. It was then cleaned up and merged together adding several other variables.
    We will use the below variables to see if we can predict victimisations with multiple linear regression.
    """)
    st.markdown('* **Police Numbers by District (Region)**')
    st.markdown('* **Household Living Costs**')
    st.markdown('* **Unemployment Rate (%)**')
    st.markdown('* **Police District**')
#sidebar
st.sidebar.header('Select Conditions')
st.sidebar.write(""" #### The Linear Regression Model """)

model = st.container()
with model:
    st.header('Linear Regression Model')
    st.write("""
    Predicting Victimisations by selecting the variables; Police numbers by district,
    Unemployment Rate, District (region), and Household living costs.
    """)


# Select available variables
available_variables = ['District_Police_Num', 'hlc_index_amt_mean', 'Unemployment rate (%)', 'Boundary_Class']

# Define a dictionary for variable renaming
variable_mapping = {
    'District_Police_Num': 'Police Numbers by District',
    'hlc_index_amt_mean': 'Household Living Costs',
    'Unemployment rate (%)': 'Unemployment Rate (%)',
    'Boundary_Class': 'Police District'
}

# Multi-select dropdown for variable selection
selected_variables = st.sidebar.multiselect('Select Variables', available_variables, format_func=lambda x: variable_mapping.get(x, x))


#Create Model

# Encode categorical variables using one-hot encoding or label encoding if necessary

# Select available variables
available_variables = ['District_Police_Num', 'hlc_index_amt_mean', 'Unemployment rate (%)', 'Boundary_Class']

# Multi-select dropdown for variable selection
#selected_variables = st.sidebar.multiselect('Select Variables', available_variables)

if len(selected_variables) > 0:
    # Filter the data based on selected variables
    X = df_crime_num_2[selected_variables]
    y = df_crime_num_2['Victimisations']

    # Interactive sliders for adjusting test size
    test_size = st.sidebar.slider("Adjust Test Size", min_value=0.1, max_value=0.5, step=0.1, value=0.3)

    # Build and train the linear regression model
    model = LinearRegression()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r_squared = model.score(X_test, y_test)

    # Get the number of samples and features
    n = X_test.shape[0]
    p = X_test.shape[1]

    # Calculate adjusted R-squared
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    # Analyze the coefficients
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})

    # Remove index column
    coefficients_without_index = coefficients.set_index('Feature')

    # Rename the feature names
    coefficients_without_index.rename(index=variable_mapping, inplace=True)

    # Print the evaluation metrics and coefficients
    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r_squared)
    st.write("Adjusted R-squared:", adjusted_r_squared)
    st.write("Coefficients:")
    st.dataframe(coefficients_without_index)



    # Scatter plot of actual vs predicted values
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolor='black')
    ax.plot(y_test, y_test, color='red', linewidth=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs Predicted Values')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add the regression line
    ax.plot(y_pred, y_pred, color='red')

    # Calculate the confidence interval
    error = y_pred - y_test
    confidence_interval = 1.96 * np.std(error)  # 95% confidence interval

    # Plot the confidence interval
    ax.fill_between(y_pred, y_pred - confidence_interval, y_pred + confidence_interval, color='gray', alpha=0.3)

    # Save the figure as a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Display the saved figure in the Streamlit app
    st.image(buffer, use_column_width=True)

else:
    st.write("Please select at least one variable.")



if len(selected_variables) > 0:

    # Calculate the residuals
    residuals = y_test - y_pred

    # Create a DataFrame for residuals
    residuals_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residuals': residuals})

    # Scatter plot of predicted values vs residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, color='blue', alpha=0.6, edgecolor='black')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals Plot')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the figure as a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Display the saved figure in the Streamlit app
    st.image(buffer, use_column_width=True)

    # Display the residuals DataFrame
    st.write("Residuals:")
    st.dataframe(residuals_df.head(5))
else:
    st.write("")

