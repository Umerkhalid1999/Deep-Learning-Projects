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

        # Identify categorical columns
        categorical_columns = selected_data.select_dtypes(include=["object", "category"]).columns.tolist()
        st.write("### Categorical Columns Identified:")
        st.write(categorical_columns)

        # Let the user specify the number of synthetic data points
        st.write("### Specify the Number of Synthetic Data Points")
        num_points = st.slider(
            "Number of synthetic data points to generate",
            min_value=1,
            max_value=10000,
            value=len(data),
        )

        # Add a button to start training
        if st.button("Train CTGAN Model"):
            st.write("### Training CTGAN Model...")
            progress_bar = st.progress(0)
            ctgan = CTGAN(epochs=50)

            # Simulate progress bar during training
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)

            try:
                # Train CTGAN
                ctgan.fit(selected_data, discrete_columns=categorical_columns)
                st.success("CTGAN Model trained successfully!")

                # Generate synthetic data
                st.write("### Generating Synthetic Data...")
                synthetic_data_raw = ctgan.sample(num_points)
                st.write(f"Raw Synthetic Data Shape: {synthetic_data_raw.shape}")

                # Ensure proper column restoration
                synthetic_data = pd.DataFrame(
                    ctgan._transformer.inverse_transform(synthetic_data_raw),
                    columns=selected_data.columns
                )

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

            except ValueError as e:
                st.error(f"An error occurred during data generation: {e}")
