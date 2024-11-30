import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


st.title('Peramalan Produksi Pupuk Urea')

# Upload file dataset dalam CSV
uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file, parse_dates=['Waktu'])
    
    # Menampilkan beberapa isi dataset
    st.subheader("Menampilkan Dataset")
    st.write(df.head(5))
    
    # # Mengecek deskriptif dataset
    st.subheader("Mengecek deskriptif dataset")
    st.write(df.describe())
    
    # #Menampilkan informasi detail tentang dataframe
    st.subheader("Menampilkan informasi detail tentang dataframe")
    st.write(df.info())
    
    #  Menampilkan nama-nama kolom
    st.write(df.columns)
    
  
    st.subheader("Grafik Produksi Urea")
    
    #Membuat Grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Waktu'], y=df['produksi'], mode='lines', name='Produksi Urea'))
    fig.update_layout(title="Produksi Pupuk UREA PT.PIM", xaxis_title='Waktu (Bulan-Tahun)', yaxis_title='Jumlah Produksi')
    st.plotly_chart(fig)

    st.title('Preprocessing')
    
    # Check for missing values
    st.subheader("Cek Nilai Yang Hilang")
    missing_values = df.isnull().sum()
    st.write(missing_values)
    
    st.subheader("Outliers")
    # Menghitung Z-Score untuk setiap nilai
    z_scores = np.abs((df['produksi'] - df['produksi'].mean()) / df['produksi'].std())
    outliers = z_scores > 3
    mask = ~outliers
    df_interpolated = df.copy()
    df_interpolated.loc[outliers, 'produksi'] = np.interp(df.index[outliers], df.index[mask], df['produksi'][mask])
    df['produksi'] = df_interpolated['produksi']
    st.write(df.head(10))
    
    # Ekstraksi Fitur
    Average = df['produksi'] / 3
    # Menambahkan fitur ke dataframe
    dataset = df.assign(Average=Average).fillna(0)
    
    dataset2 = dataset.sort_values(by=['Waktu']).copy()
    fitur = ['produksi', 'Average']
    st.write('Macam Macam Fitur:', fitur)
    
    # Membuat dataset sesuai dengan list pada fitur
    data2 = pd.DataFrame(dataset2)
    dataset3 = data2[fitur]
    
    # Banyaknya baris
    n_baris = dataset3.shape[0]

    # Convert data dalam bentuk array
    np_data_unscaled = np.array(dataset3)
    np_data = np.reshape(np_data_unscaled, (n_baris, -1))
    st.write('Banyak Baris dan Kolom', np_data.shape)
    
    # Fungsi untuk segmentasi data
    def partition_dataset(sequence_length, data):
        x, y = [], []
        for i in range(len(data) - sequence_length):
            x.append(data[i:i+sequence_length, :])
            y.append(data[i+sequence_length, 0])
        return np.array(x), np.array(y)

    # Tentukan panjang urutan untuk segmentasi
    panjang_urutan = 12

    
    # Normalize the dataset
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(dataset3)
    df_normalized = pd.DataFrame(np_data_scaled, columns=dataset3.columns)

    
    st.subheader("Data Latih dan Data Uji")
    # Split data latih dan data uji Produksi Urea
    n_datalatih = 48
    n_datauji = 12
    train_data = np_data_scaled[:n_datalatih, :]
    test_data = np_data_scaled[-n_datauji:, :]
    # Print the number of rows in training and testing data
    st.write(f'Total baris untuk data Latih: {train_data.shape[0]}')
    st.write(f'Total baris untuk data Uji: {test_data.shape[0]}')

    # Fungsi untuk segmentasi data
    def partition_dataset(sequence_length, data):
        x, y = [], []
        for i in range(len(data) - sequence_length):
            x.append(data[i:i+sequence_length, :])
            y.append(data[i+sequence_length, :])
        return np.array(x), np.array(y)
    # Tentukan panjang urutan untuk segmentasi
    panjang_urutan = 12

    # Segmentasi data pelatihan dan pengujian
    x_train, y_train = partition_dataset(panjang_urutan, train_data)
    x_test, y_test = partition_dataset(panjang_urutan, test_data)
    
    # Split data untuk visualisasi
    train = dataset3['produksi'][:n_datalatih + 1]
    test = dataset3['produksi'][n_datalatih:]
    # Plot data latih dan uji
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Testing', line=dict(color='orange')))
    fig.update_layout(title='Total Produksi Pupuk Urea \n80:20 Data Latih & Data Uji', xaxis_title='Total Data', yaxis_title='Total Produksi')
    st.plotly_chart(fig)
    
    st.title('Weighted Moving Average(WMA)')
    ## Weighted Moving Average (WMA)
    weights = [0.7, 0.2, 0.1]
    #Membangun Model
    def calculate_wma(data, weights):
        wma = [np.nan]
        for i in range(len(data)):
            if i >= len(weights)-1:
                weighted_sum = sum(data[i-j] * weights[j] for j in range(len(weights)))
                wma.append(weighted_sum)
            else:
                wma.append(np.nan)
        return wma
    # Hitung WMA untuk data train
    wma_train = calculate_wma(train, weights)
    
    # Peramalan WMA untuk data tes
    st.subheader("Hasil peramalan menggunakan data Latih")
    def forecast_wma(data, weights, months=12):
        forecasts = []
        for _ in range(months):
            if len(data) >= len(weights):
                forecast = sum(data[-j-1] * weights[j] for j in range(len(weights)))
                forecasts.append(forecast)
                data = np.append(data, forecast)
            else:
                forecasts.append(np.nan)
        return forecasts
    
    forecasts = forecast_wma(wma_train, weights)
    #Membuat Tabel untuk Hasil peramalan
    forecast_table = pd.DataFrame({
        'Waktu': df['Waktu'][-12:],
        'test_WMA' : test,
        'Forecast_train': forecasts
    })
    st.write(forecast_table)
    
    # Membuat objek gambar dan sumbu
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot Data Uji
    ax.plot(forecast_table['Waktu'], forecast_table['test_WMA'], label='Actual_Train', marker='o')
    # Plot Hasil peramalan dari data latih
    ax.plot(forecast_table['Waktu'], forecast_table['Forecast_train'], label='Test', marker='s')
    # Menambahkan label data ke setiap titik
    for i, row in forecast_table.iterrows():
        ax.annotate(f" {row['test_WMA']:.2f}", (row['Waktu'], row['test_WMA']), textcoords="offset points", xytext=(0, 5), ha='center')
        ax.annotate(f"{row['Forecast_train']:.2f}", (row['Waktu'], row['Forecast_train']), textcoords="offset points", xytext=(0, -15), ha='center')
    # Membuat Label Grasik
    ax.set_title('Perbandingan Hasil Peramalan Data Latih dengan Data Uji (Produksi Urea)')
    ax.set_xlabel('Waktu')
    ax.set_ylabel('Value')
    # Legenda
    ax.legend()
    # Show the plot
    st.subheader("Grafik Perbandingan Hasil Peramalan Data Latih dengan Data Uji")
    st.pyplot(fig)
    
    st.subheader("Perhitungan WMA Untuk Seluruh Data Produksi Urea")
    # Pemberian bobot
    weights = [0.7, 0.2, 0.1]
    # membuat model perhitungan WMA
    def calculate_wma(data, weights):
        wma = [np.nan]  
        for i in range(len(data)):
            if i >= len(weights)-1:  
                weighted_sum = sum(data[i-j] * weights[j] for j in range(len(weights)))
                wma.append(weighted_sum)
            else:
                wma.append(np.nan)  # Append NaN for the first few elements
        return wma
    produksi_urea = df['produksi'].values
    # Melakukan perhitungan WMA
    wma_values = calculate_wma(produksi_urea, weights)
    # Ambil data produksi amonia
    produksi_urea = df['produksi'].values
    # Hitung WMA
    wma_values = calculate_wma(produksi_urea, weights)
    # Membuat DataFrame dengan data asli dan nilai WMA
    df_wma = pd.DataFrame({'Produksi UREA': x, 'wma_produksi_urea': y} for x, y in zip(produksi_urea, wma_values[:]))
    st.write(df_wma)
    #hilangkan 3 baris pertama pada data yang ada nilai Nan untuk dapat melakukan uji akurasi
    df_wma = df_wma.iloc[3:]
    
    st.title('Evaluasi Model')
    st.subheader("Pengujian Akurasi dengan MAPE dan MAD")
    
    #membuat perhitungan MAPE
    mape = (np.abs(df_wma['Produksi UREA'] - df_wma['wma_produksi_urea']) / df_wma['Produksi UREA']) * 100
    # mengubah nilai NaN dan inf menjadi 0
    mape = mape.replace([np.inf, -np.inf], 0).fillna(0)
    # menghitung average MAPE
    avg_mape = np.mean(mape)
    st.write(f"MAPE: {avg_mape:.1f}%")
    
    # membuat perhitungan MAD
    mad = (np.abs(df_wma['Produksi UREA'] - df_wma['wma_produksi_urea']))
    # mengubah nilai NaN dan inf menjadi 0
    mad = mad.replace([np.inf, -np.inf], 0).fillna(0)
    # menghitung average MAD
    avg_mad = np.mean(mad)
    st.write(f" MAD: {avg_mad:.1f}")
    #Membuat Tabel Untuk semua perhitungan
    st.subheader("Tabel Hasil Perhitungan WMA Produksi Urea")
    tabel_akurasi = pd.DataFrame({
        'produksi_urea': dataset3['produksi'],
        'wma_produksi_urea': df_wma['wma_produksi_urea'],
        'MAPE': mape,
        'MAD': mad
    })
    st.write(tabel_akurasi)
    
    st.title('Peramalan 12 bulan kedepan')
    # pembentukan perhitungan peramalan untuk produksi amonia
    def forecast_wma(data, weights, months=12):
        forecasts = []
        data_values = data.values # Ekstrak nilai sebagai values NumPy
        for _ in range(months):
            if len(data_values) >= len(weights):
                forecast = sum(data_values[-j-1] * weights[j] for j in range(len(weights)))
                forecasts.append(forecast)
                data_values = np.append(data_values, forecast)
            else:
                forecasts.append(np.nan)
        return forecasts
    # Melakukan perhitungan peramalan
    forecasts12 = forecast_wma(df_wma['Produksi UREA'], weights)
    #Mengubah hasil peramalan dalam bentuk DataFrame
    forecasts12_df = pd.DataFrame({'Peramalan_amonia': forecasts12})
    forecasts12_df['waktu'] = pd.date_range(start='2024-01-01', periods=12, freq='M')
    forecasts12_df['waktu'] = forecasts12_df['waktu'].apply(lambda x: x.replace(day=1))
    forecasts12_df = forecasts12_df.set_index('waktu').reset_index()
    st.write(forecasts12_df)
    

    # Plot Original Data vs Forecast
    fig = go.Figure()
    # menambahkan keterangan
    fig.add_trace(go.Scatter(x=df['Waktu'], y=df['produksi'], mode='lines', name='Original', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecasts12_df['waktu'], y=forecasts12_df['Peramalan_amonia'], mode='lines', name='Forecast', line=dict(color='orange')))
    # Perbarui tata letak
    fig.update_layout(title='Grafik Peramalan Produksi Pupuk Urea 12 bulan',
                        xaxis_title='Waktu', yaxis_title='Produksi Pupuk Amonia', xaxis=dict(tickformat='%Y-%m-%d',
                                                                                        tickmode='auto', nticks=20))
    # Tampilkan Grafik
    st.plotly_chart(fig)