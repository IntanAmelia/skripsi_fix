# app.py
#import library

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import math
import joblib
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model

# Set the title of the app
st.title("PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM")

# Add a sidebar title
st.sidebar.title("Main Menu")

menu = st.sidebar.radio("Go to", ["Dataset", "Deteksi Outlier dan Interpolasi Linear", "Normalisasi Data", "Model LSTM", "Prediksi LSTM", "Implementasi"])

if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_imputed' not in st.session_state:
    st.session_state.df_imputed = None
# Add different sections based on the selected menu item
if menu == "Dataset":
    st.write("""
    <h5>Data Understanding</h5>
    <br>
    """, unsafe_allow_html=True)

    st.write('Dataset ini berisi data tentang curah hujan. Dataset yang digunakan pada penelitian ini berasal dari website https://dataonline.bmkg.go.id berdasarkan hasil pengamatan Badan Meteorologi, Klimatologi, dan Geofisika Stasiun Meteorologi Maritim Tanjung Perak dari 1 Januari 2019 hingga 31 Agustus 2023.')
    missing_values = ['8888']
    df = pd.read_excel('Dataset_Curah_Hujan.xlsx', na_values = missing_values)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
    st.session_state.df = df
    st.write("Dataset Curah Hujan : ")
    st.write(df)
elif menu == "Deteksi Outlier dan Interpolasi Linear":
    df = st.session_state.df
    if df is not None:
        is_outlier_iqr = (df['RR'] < 0) | (df['RR'] > 100)
        outliers = is_outlier_iqr
        df['Outlier'] = outliers
        st.session_state.df = df
        st.write('Dataset yang termasuk outlier:')
        st.dataframe(df[['RR', 'Outlier']])
            
        # Replace outliers with linear interpolation values
        data_cleaned = df['RR'].copy()
        for i, is_outlier in enumerate(outliers):
            if is_outlier:
                data_cleaned[i] = (df['RR'].iloc[i-1] + df['RR'].iloc[i+1]) / 2
        df['interpolasi outlier'] = data_cleaned
        st.session_state.df = df
        st.write('Data setelah dilakukan interpolasi :')
        st.dataframe(df[['RR', 'interpolasi outlier']])
    else:
        st.write('Silahkan memasukkan dataset terlebih dahulu.')
elif menu == "Normalisasi Data":
    df = st.session_state.df
    if df is not None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        st.session_state.scaler = scaler
        scaled_data = scaler.fit_transform(df['interpolasi outlier'].values.reshape(-1,1))
        df['Normalisasi'] = scaled_data
        st.session_state.df = df
        st.session_state.scaled_data = scaled_data
        st.write('Data setelah dilakukan normalisasi :')
        st.write(df[['interpolasi outlier', 'Normalisasi']])
    else:
        st.write('Silahkan melakukan deteksi outlier terlebih dahulu')
elif menu == "Model LSTM":
    df = st.session_state.df
    scaler = st.session_state.scaler
    scaled_data = st.session_state.scaled_data
    if df is not None and scaler is not None and scaled_data is not None:
        epochs = st.number_input("Masukkan nilai epoch:", min_value=1, max_value=100, value=100)
        learning_rate = st.number_input("Masukkan nilai learning rate:", min_value=0.0001, max_value=0.01, value=0.01)
        time_steps = st.number_input("Masukkan nilai time step:", min_value=25, max_value=75, value=25)
        split_data = st.number_input("Masukkan nilai data train:", min_value=0.5, max_value=0.9, value=0.9)
        st.session_state.time_steps = time_steps
    
        if st.button('Simpan'):
            values = scaled_data
            training_data_len = math.ceil(len(values) * 0.7)
            st.session_state.training_data_len = training_data_len
            train_data = scaled_data[:training_data_len]

            x_train = []
            y_train = []

            for i in range(time_steps, len(train_data)):
                x_train.append(train_data[i - time_steps:i, 0])
                y_train.append(train_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            test_data = scaled_data[training_data_len - time_steps:]
            x_test = []
            y_test = []

            for i in range(time_steps, len(test_data)):
                x_test.append(test_data[i - time_steps:i, 0])
                y_test.append(test_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            st.session_state.x_train = x_train
            st.session_state.x_test = x_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
    
            def build_and_train_lstm(x_train, y_train, x_test, y_test, epochs, learning_rate):
                model = Sequential()
                model.add(LSTM(100, name='lstm_layer_1', return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(Dropout(0.5))
                model.add(LSTM(100, name='lstm_layer_2'))
                model.add(Dropout(0.5))
                model.add(Dense(1))
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mean_squared_error')
                model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_test, y_test))
                st.session_state.model = model
                return model
    
            model = build_and_train_lstm(x_train, y_train, x_test, y_test, epochs, learning_rate)
            st.session_state.model = model
            st.write("Model telah disimpan dan dilatih.")
    else:
        st.write('Silahkan melakukan proses normalisasi data terlebih dahulu.')
elif menu == "Prediksi LSTM":
    if st.session_state.x_train is not None and st.session_state.x_test is not None and st.session_state.y_train is not None and st.session_state.y_test is not None and st.session_state.model is not None and st.session_state.scaler is not None and st.session_state.scaled_data is not None and st.session_state.training_data_len is not None and st.session_state.time_steps is not None:
        test_predictions = st.session_state.model.predict(st.session_state.x_test)
        test_predictions_data = st.session_state.scaler.inverse_transform(test_predictions)
        data_prediksi_uji = pd.DataFrame(test_predictions_data, columns=['Hasil Prediksi Data Uji'])
        st.session_state.data_prediksi_uji = data_prediksi_uji
        data_asli = st.session_state.df['RR'][-len(st.session_state.x_test):].to_numpy()
        rmse = np.sqrt(np.mean((data_asli - test_predictions_data) ** 2))
        st.write('Hasil Prediksi Data Uji:')
        st.write(data_prediksi_uji)
        st.write('RMSE Data Uji')
        st.write(rmse)
        plt.figure(figsize=(20, 7))
        plt.plot(st.session_state.df['Tanggal'][-len(st.session_state.x_test):], st.session_state.df['RR'][-len(st.session_state.x_test):], color='blue', label='Curah Hujan Asli')
        plt.plot(st.session_state.df['Tanggal'].iloc[-len(data_prediksi_uji):], data_prediksi_uji['Hasil Prediksi Data Uji'], color='red', label='Prediksi Curah Hujan')
        plt.title('Prediksi Curah Hujan')
        plt.xlabel('Tanggal')
        plt.ylabel('Curah Hujan (mm)')
        plt.legend()
        # Menampilkan plot di Streamlit
        st.pyplot(plt)
    else:
        st.write('Silahkan bangun model terlebih dahulu')
elif menu == "Implementasi":
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test
    model = st.session_state.model
    scaler = st.session_state.scaler
    df = st.session_state.df
    data_prediksi_uji = st.session_state.data_prediksi_uji
    time_steps = st.session_state.time_steps
    if x_test is not None and model is not None and scaler is not None and df is not None and data_prediksi_uji is not None and time_steps is not None and y_test is not None:
        n = st.selectbox("Pilih prediksi selanjutnya :", [1, 2, 7, 14, 30, 180, 365])
        future_predictions = []
        x_last_window = np.array(x_test[-time_steps:], dtype=np.float32).reshape((1, -1, 1))
        y_test_scaler = st.session_state.scaler.inverse_transform(st.session_state.y_test.reshape(-1, 1))
        for _ in range(n):
            # Predict the next time step
            prediction = model.predict(x_last_window)
            # Append the prediction to the list of future predictions
            future_predictions.append(prediction[0])
            
            # Update the last window by removing the first element and appending the prediction
            x_last_window = np.append(x_last_window[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
            
        # Convert the list of future predictions to a numpy array
        future_predictions = np.array(future_predictions)
        future_predictions = future_predictions.round(2)
            
        # Inverse transform predictions to get the original scale
        future_predictions_denormalisasi = scaler.inverse_transform(future_predictions)
        future_predictions_denormalisasi = future_predictions_denormalisasi.round(2)
        future_predictions_df = pd.DataFrame(future_predictions_denormalisasi, columns=['Prediksi'])
        st.write('Prediksi Selanjutnya : ')
        st.write(future_predictions_df)
            
        # Plotting the predictions
        plt.figure(figsize=(12, 6))
        future_dates = pd.date_range(start=df['Tanggal'].iloc[-1], periods=n+1, closed='right')
        if n == 1:
            plt.plot(future_dates, future_predictions_df, 'ro', label='Prediksi Selanjutnya')
        else:
            plt.plot(future_dates, future_predictions_df, color='red', label='Prediksi Selanjutnya')
            
        plt.title('Prediksi Curah Hujan Selanjutnya')
        plt.xlabel('Tanggal')
        plt.ylabel('Curah Hujan (mm)')
        plt.legend()
        st.pyplot(plt)    
    else:
        st.write('Silahkan melakukan prediksi terlebih dahulu')
