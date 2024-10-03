import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Load model, scaler and encoder
scaler = joblib.load('scaler_final.pkl')
stacking_model = joblib.load('stacking_model.pkl')
column_order = joblib.load('column_order.pkl') 

# Load encoders
categorical_columns = ['conditions_arrival', 'conditions_departure']

label_encoders = {}
for col in categorical_columns:
    try:
        label_encoders[col] = joblib.load(f'{col}_label_encoder.pkl')  
    except FileNotFoundError:
  
        st.error(f"Encoder not found for column : {col}")

# Function to add predictions, days and conditions to the session
def ajouter_pred(prediction, weekday, conditions_arrival, conditions_departure):
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []
        st.session_state['days'] = []
        st.session_state['conditions'] = []  #conditions Initialisation

    # Add values to session_state
    st.session_state['predictions'].append(prediction)
    st.session_state['days'].append(weekday)
    st.session_state['conditions'].append({
        'conditions_arrival': conditions_arrival,
        'conditions_departure': conditions_departure
    })

# Page 1: Make prediction
def make_prediction():
    st.markdown("<h1 style='color: #4CAF50;'>Flight Delay Prediction Application</h1>", unsafe_allow_html=True)
    st.write("This application predicts whether a flight will be delayed or not, based on weather conditions, flight duration, day of the week...")

    # Mapping between days of the week and their numerical values
    days_of_week_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    # Collect user input for the day of the week
    weekday_str = st.selectbox("Day of the Week", list(days_of_week_mapping.keys()))
    weekday = days_of_week_mapping[weekday_str]

    # Collect user data for weather conditions
    conditions_arrival = st.selectbox("Weather Condition at Arrival", ['Overcast', 'Partially cloudy', 'Rain, Overcast', 'Rain, Partially cloudy', 'Snow, Rain, Overcast', 'Snow, Rain, Partially cloudy'])
    conditions_departure = st.selectbox("Weather Condition at Departure", ['Overcast', 'Partially cloudy', 'Rain, Overcast', 'Rain, Partially cloudy', 'Snow, Rain, Overcast', 'Snow, Rain, Partially cloudy'])

    # Collect other user data
    input_data = pd.DataFrame({
        'tempmax_arrival': [st.number_input("Max Temperature at Arrival (°C)", min_value=2.33, max_value=14.88, value=10.94)],
        'tempmin_arrival': [st.number_input("Min Temperature at Arrival (°C)", min_value=-1.61, max_value=11.39, value=7.61)],
        'humidity_arrival': [st.number_input("Humidity at Arrival (%)", min_value=73.6, max_value=92.5, value=78.8)],
        'conditions_arrival': [conditions_arrival],
        'tempmax_departure': [st.number_input("Max Temperature at Departure (°C)", min_value=2.33, max_value=14.88, value=10.94)],
        'tempmin_departure': [st.number_input("Min Temperature at Departure (°C)", min_value=-1.61, max_value=11.39, value=7.61)],
        'humidity_departure': [st.number_input("Humidity at Departure (%)", min_value=73.6, max_value=92.5, value=78.8)],
        'conditions_departure': [conditions_departure],
        'flight_duration': [st.number_input("Flight Duration (hours)", min_value=0.1, max_value=10.0, value=1.0) * 3600],
        'weekday': [weekday]
    })

    # Encode categorical columns
    for col in categorical_columns:
        if col in input_data.columns and col in label_encoders:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))  # Transform with encoder loaded
            except ValueError as e:
                st.error(f"Error when encoding '{col}': {e}")

    # Separate numeric and categorical columns
    numerical_columns = ['tempmax_arrival', 'tempmin_arrival', 'humidity_arrival',
                         'tempmax_departure', 'tempmin_departure', 'humidity_departure',
                         'flight_duration', 'weekday']

    # Scale numeric columns
    input_data_numerical_scaled = scaler.transform(input_data[numerical_columns])

    # Combine scaled and categorical columns
    input_data_scaled = pd.DataFrame(input_data_numerical_scaled, columns=numerical_columns)
    input_data_scaled[categorical_columns] = input_data[categorical_columns].values

    # Rearrange columns according to the order used during training
    input_data_scaled = input_data_scaled.reindex(columns=column_order)

    # Make prediction
    if st.button("Predict Delay"):
        try:
            prediction = stacking_model.predict(input_data_scaled)

            # Show result
            if prediction[0] == 1:
                st.markdown("<h3 style='color: #f44336;'>The flight is predicted to be late.</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: #4CAF50;'>Flight predicted on time.</h3>", unsafe_allow_html=True)

            # Add prediction and weather conditions to history
            ajouter_pred(prediction[0], weekday, conditions_arrival, conditions_departure)
        except ValueError as e:
            st.error(f"Erreur lors de la prédiction: {e}")



# Page 2: Delay statistics
def delay_statistics2():
    st.markdown("<h1 style='color: #4CAF50;'>Statistics on Delayed Flights</h1>", unsafe_allow_html=True)

    total_predictions = len(st.session_state.get('predictions', []))
    if total_predictions == 0:
        st.write("No predictions yet.")
        return

    # Calculate on-time and delayed flights
    delayed = sum(st.session_state['predictions'])  # Flights predicted as delayed
    on_time = total_predictions - delayed  # Flights on time

    # Improved display using HTML for styling
    st.markdown(f"""
        <div style="font-size:18px;">
            <p><strong>Total number of flights analyzed :</strong> {total_predictions}</p>
            <p style="color:green;"><strong>Flights on time :</strong> {on_time} ({on_time / total_predictions * 100:.2f}%)</p>
            <p style="color:red;"><strong>Delayed flights :</strong> {delayed} ({delayed / total_predictions * 100:.2f}%)</p>
        </div>
    """, unsafe_allow_html=True)





# Page 3: Best Day to Travel
def display_best_day2():
    st.markdown("<h1 style='color: #4CAF50;'>Best Day to Travel</h1>", unsafe_allow_html=True)

    if len(st.session_state.get('predictions', [])) == 0:
        st.write("No predictions yet.")
        return

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    delays_per_day = {day: {'delay': 0, 'total': 0} for day in days}

    # Calculate delays by day
    for pred, day in zip(st.session_state['predictions'], st.session_state['days']):
        day_name = days[day]
        delays_per_day[day_name]['total'] += 1
        if pred == 1:  # If the flight is delayed
            delays_per_day[day_name]['delay'] += 1

    # Calculate the delay rate for each day
    delay_rate = {day: (info['delay'] / info['total'] * 100) if info['total'] > 0 else 0
                  for day, info in delays_per_day.items()}

    # Display the day with the fewest delays
    best_day = min(delay_rate, key=delay_rate.get)
    st.write(f"**The best day to travel is: {best_day}** with a delay rate of only {delay_rate[best_day]:.2f}%.")

    # Convert the dictionary to a DataFrame for plotting
    delay_rate_df = pd.DataFrame(list(delay_rate.items()), columns=['Day', 'Delay Rate (%)'])

    # Display the delay rates using a horizontal bar chart
    st.subheader("Delay Rate by Day of the Week")
    fig, ax = plt.subplots()
    ax.barh(delay_rate_df['Day'], delay_rate_df['Delay Rate (%)'], color='skyblue')
    ax.set_xlabel('Delay Rate (%)')
    ax.set_title('Delay Rate by Day of the Week')

    # Display the chart in Streamlit
    st.pyplot(fig)

# Sidebar pnavigation
st.sidebar.markdown("<h2 style='color: #007BFF;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.selectbox(
    "Go To", 
    ("Make a prediction", "Delay statistics", "Best day to travel"), 
)

# Display pages 
if page == "Make a prediction":
    make_prediction()  
elif page == "Delay statistics":
    delay_statistics2()
elif page == "Best day to travel":
    display_best_day2()


