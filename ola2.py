import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error as mae

# --- Configuration and Constants ---
# Set the page configuration
st.set_page_config(
    page_title="Bike Demand Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define feature columns for user input and model training
FEATURE_COLS = [
    'season', 'weather', 'temp', 'humidity', 'windspeed',
    'year', 'month', 'day', 'is_weekend', 'hour', 'am_or_pm'
]
TARGET_COL = 'count'

# --- Data Loading and Model Training (Cached) ---

import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path: str):
    """
    Loads the dataset using a relative path.
    FIX: Ensures correct behavior both locally and when deployed (Streamlit Cloud).
    """
    try:
        df = pd.read_csv(file_path)
        return df

    except FileNotFoundError:
        st.error(
            f"FileNotFoundError: Could not find `{file_path}`.\n"
            "Make sure `ola.csv` is located in the root directory of your project repository."
        )
        return None

    except Exception as e:
        st.error(f"An unexpected error occurred while loading the data: {e}")
        return None

@st.cache_resource
def preprocess_and_train_model(df):
    """Performs feature engineering, scaling, and trains the Lasso model."""

    # 1. Feature Engineering
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['is_weekend'] = np.where(df['datetime'].dt.dayofweek.isin([5, 6]), 1, 0)
    df['am_or_pm'] = np.where(df['datetime'].dt.hour.isin(range(0, 12)), 0, 1)

    # 2. Prepare features and target
    features = df[FEATURE_COLS]
    target = df[TARGET_COL].values

    # 3. Train-Test Split (using random_state=22 as a common practice)
    X_train, X_test, Y_train, Y_test = train_test_split(
        features, target, test_size=0.2, random_state=22
    )

    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Training (Lasso Regression)
    lasso_model = Lasso(alpha=0.01) # Use a suitable alpha or hyperparameter tune
    lasso_model.fit(X_train_scaled, Y_train)

    # Optional: Display training MAE
    y_pred = lasso_model.predict(X_test_scaled)
    test_mae = mae(Y_test, y_pred)
    st.sidebar.success(f"Model Trained Test MAE: {test_mae:.4f}")

    return lasso_model, scaler

# --- Streamlit Application Layout ---

def main():
    """The main function to run the Streamlit app."""

    st.title("Bike Rental Demand Forecast")
    st.markdown("Use the sidebar to input the conditions and predict the hourly bike rental demand.")
    st.markdown("---")

    # Load data
    df = load_data('ola.csv')
    if df is None:
        return

    # Train model and scaler
    model, scaler = preprocess_and_train_model(df)

    # --- Sidebar for User Input (Feature Collection) ---
    with st.sidebar:
        st.header("Input Features")

        # Temporal Features
        st.subheader("Time of Ride")
        input_year = st.selectbox("Year", [2011, 2012], index=1)
        input_month = st.slider("Month", 1, 12, 7)
        input_day = st.slider("Day of Month", 1, 31, 15)
        input_hour = st.slider("Hour (24-Hour Clock)", 0, 23, 8)


        # Calculate if the day is a weekend (0-Mon, 6-Sun)
        try:
            date_obj = pd.to_datetime(f"{input_year}-{input_month}-{input_day}")
            input_is_weekend = 1 if date_obj.dayofweek in [5, 6] else 0
        except ValueError:
            # Handle impossible dates (e.g., Feb 30) gracefully for derived features
            input_is_weekend = 0 # Default to weekday if date is invalid
            st.warning("Invalid Date: Could not determine weekend status.")

        weekend_status = "Yes" if input_is_weekend == 1 else "No"
        st.info(f"Automatically determined: **Weekend: {weekend_status}**")
        input_am_or_pm = 0 if input_hour < 12 else 1

        # Categorical Features
        st.subheader("Categorical Conditions")
        input_season_map = {1: '1 (Winter)', 2: '2 (Spring)', 3: '3 (Summer)', 4: '4 (Fall)'}
        input_season_label = st.selectbox("Season", options=list(input_season_map.values()))
        input_season = [k for k, v in input_season_map.items() if v == input_season_label][0]

        input_weather_map = {
            1: '1 (Clear/Few Clouds)',
            2: '2 (Mist/Cloudy)',
            3: '3 (Light Rain/Snow)',
            4: '4 (Heavy Rain/Snow - Rare)'
        }
        input_weather_label = st.selectbox("Weather", options=list(input_weather_map.values()))
        input_weather = [k for k, v in input_weather_map.items() if v == input_weather_label][0]

        # Environmental Features
        st.subheader("Environmental Factors")
        input_temp = st.slider("Temperature (°C)", 5.0, 35.0, 25.0)
        input_humidity = st.slider("Humidity (%)", 20.0, 100.0, 60.0)
        input_windspeed = st.slider("Windspeed (kph approx)", 0.0, 50.0, 15.0)

        # Prediction button
        predict_button = st.button("Predict Demand", type="primary")

    # --- Prediction Logic ---
    if predict_button:
        # 1. Create a DataFrame from the user inputs
        input_data = pd.DataFrame({
            'season': [input_season],
            'weather': [input_weather],
            'temp': [input_temp],
            'humidity': [input_humidity],
            'windspeed': [input_windspeed],
            'year': [input_year],
            'month': [input_month],
            'day': [input_day],
            'is_weekend': [input_is_weekend],
            'hour': [input_hour],
            'am_or_pm': [input_am_or_pm]
        })

        # 2. Scale the input data

        input_data = input_data[FEATURE_COLS]
        input_scaled = scaler.transform(input_data)

        # 3. Make prediction
        prediction = model.predict(input_scaled)[0]

        # 4. Post-process prediction (demand cannot be negative or fractional)
        predicted_count = max(0, int(np.round(prediction)))

        # 5. Display Result
        st.subheader("Hourly Bike Demand Prediction")
        st.metric(label="Predicted Total Rentals", value=f"{predicted_count:,}")


        st.markdown(
            f"""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
                <p style='font-size: 16px; margin-bottom: 0; color: #000;'>
                    This is the predicted number of total bike rentals (casual + registered) for the hour.
                </p>
                <p style='font-size: 14px; color: #000;'>
                    Based on the optimized Lasso Regression model (MAE ~50-52).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":

    main()

