import streamlit as st
import pandas as pd
from ctgan import CTGAN
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Title of the app
st.title("Flexible Synthetic Data Generator")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Original Data")
    st.write(data.head())

    # Select columns for synthetic data generation
    st.write("### Select Columns for Synthetic Data Generation")
    columns = st.multiselect("Choose columns", data.columns)

    if columns:
        # Preprocess the selected columns
        selected_data = data[columns]

        # Let the user specify the number of synthetic data points
        st.write("### Specify the Number of Synthetic Data Points")
        num_points = st.slider(
            "Number of synthetic data points to generate",
            min_value=1,
            max_value=10000,  # Adjust max_value as needed
            value=len(data),  # Default to the size of the original dataset
        )

        # Add a button to start training
        if st.button("Train CTGAN Model"):
            # Train CTGAN model
            st.write("### Training CTGAN Model...")
            progress_bar = st.progress(0)  # Initialize progress bar
            status_text = st.empty()  # Placeholder for status updates

            # Simulate progress during training
            for i in range(100):
                time.sleep(0.1)  # Simulate training time
                progress_bar.progress(i + 1)
                status_text.text(f"Training progress: {i + 1}%")

            ctgan = CTGAN(epochs=100)  # Adjust epochs as needed
            ctgan.fit(selected_data, discrete_columns=selected_data.select_dtypes(include=['object', 'category']).columns)

            # Generate synthetic data
            st.write("### Generating Synthetic Data...")
            progress_bar.progress(0)  # Reset progress bar for data generation
            status_text.text("Generating synthetic data...")

            synthetic_data = ctgan.sample(num_points)  # Generate the specified number of rows

            # Simulate progress during data generation
            for i in range(100):
                time.sleep(0.03)  # Simulate data generation time
                progress_bar.progress(i + 1)
                status_text.text(f"Data generation progress: {i + 1}%")

            st.write("### Synthetic Data")
            st.write(synthetic_data.head())

            # Visualize comparisons
            st.write("### Comparison of Original vs Synthetic Data")
            for column in columns:
                plt.figure(figsize=(10, 5))
                sns.histplot(selected_data[column], label='Original Data', kde=True, color='blue')
                sns.histplot(synthetic_data[column], label='Synthetic Data', kde=True, color='orange')
                plt.legend()
                plt.title(f'Comparison of {column} Distributions')
                st.pyplot(plt)

            # Download synthetic data
            st.write("### Download Synthetic Data")
            synthetic_csv = synthetic_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Synthetic Data as CSV",
                data=synthetic_csv,
                file_name='synthetic_data.csv',
                mime='text/csv',
            )

