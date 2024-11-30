import streamlit as st

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

def main():
    # Dataframe selection
    st.markdown("<h1 align='center'> <b>OPTIMASI KETAHANAN PANGAN PERAMALAN PRODUKSI PUPUK DENGAN METODE WEIGHTED MOVING AVERAGE (WMA)</b></h1>", unsafe_allow_html=True)
    new_line(1)
    st.markdown("Selamat datang! Aplikasi ini dirancang untuk menganalisis kinerja metode Weighted Moving Average (WMA) dalam meramalkan produksi pupuk guna mendukung optimasi ketahanan pangan. Sistem ini diharapkan dapat menjadi landasan dalam pengambilan keputusan yang lebih tepat terkait produksi dan distribusi pupuk, sehingga berkontribusi pada keberlanjutan ketahanan pangan.", unsafe_allow_html=True)
    
    st.divider()
    
    # Overview
    new_line()
    st.markdown("<h2 style='text-align: center;'>🗺️ Gambaran Umum</h2>", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""
    Dalam proses memcbangun model prediksi, ada serangkaian langkah yang harus diikuti. Berikut ini adalah langkah-langkah utama dalam proses Machine Learning:
    
    - **📦 Pengumpulan Data**: Proses pengumpulan data dilakukan pada PT. Pupuk Iskandar (PIM) berupa bentuk tabel excel.<br> <br>
    - **🧹 Pembersihan Data**: Proses pembersihan data dengan menghapus duplikasi, menangani nilai yang hilang, serta mengidentifikasi dan menangani outlier. Langkah ini sangat penting karena data sering kali tidak bersih dan memerlukan perbaikan untuk memastikan keakuratan dan kualitas analisis.<br> <br>
    - **⚙️ Pra-pemrosesan Data**: Proses mengubah data ke dalam format yang sesuai untuk analisis. Ini termasuk menangani fitur kategorikal, fitur numerik, penskalaan dan transformasi, dll.<br> <br>
    - **💡 Rekayasa Fitur**: Proses yang memanipulasi fitur itu sendiri. Terdiri dari beberapa langkah seperti ekstraksi fitur, transformasi fitur, dan pemilihan fitur.<br> <br>
    - **✂️ Pembagian Data**: Proses membagi data menjadi set pelatihan dan pengujian. Set pelatihan digunakan untuk melatih model dan set pengujian digunakan untuk mengevaluasi model.<br> <br>
    - **🧠 Membangun Model Pembelajaran Mesin**: Model yang digunakan pada aplikasi ini adalah Weighted Moving Average (WMA).<br> <br>
    - **⚖️ Evaluasi Model Pembelajaran Mesin**: Proses mengevaluasi model prediksi dengan menggunakan metrik seperti Mean Absolute Percentage Error (MAPE) dan Mean Absolute Devisasion (MAD).<br> <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Pada bagian membangun model, pembobotan dilakukan dengan menggunakan bobot:
    
    - **0,7**: Satu bulan yang lalu
    - **0,2**: Dua bulan yang lalu
    - **0,1**: Tiga Bulan Lalu
    """, unsafe_allow_html=True)
    new_line()
    
    # Source Code
    new_line()
    st.header("📂 Source Code")
    st.markdown("Untuk pengembangan aplikasi ini, source code tersedia di [**GitHub**](https://github.com/hayuraaa/Forecasting-LSTM-GRU.git). Jangan ragu untuk berkontribusi, memberikan feedback, atau menyesuaikan aplikasi agar sesuai dengan kebutuhan Anda.", unsafe_allow_html=True)
    new_line()
    
    # Contributors
    st.header("👤 Kontributor")
    st.markdown("Aplikasi dibuat untuk kebutuhan tugas akhir/skripsi, **Rifkial Iqwal** (200170175) .", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""Jika Anda memiliki pertanyaan atau saran, jangan ragu untuk menghubungi **rifkialikhwal@gmail.com**. Kami siap membantu!

**Connect with us on social media:** 

<a href="https://www.linkedin.com/in/adiprrassetyo/" target="_blank">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe0adDoUGWVD3jGzfT8grK5Uhw0dLXSk3OWJwZaXI-t95suRZQ-wPF7-Az6KurXDVktV4&usqp=CAU" alt="LinkedIn" width="80" height="80" style="border-radius: 25%;">
</a>       
<a href="https://www.instagram.com/adiprrassetyo/" target="_blank">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/2048px-Instagram_logo_2016.svg.png" alt="Instagram" width="80" height="80" style="border-radius: 25%;">
</a>       
<a href="https://github.com/adiprrassetyo/" target="_blank">
  <img src="https://seeklogo.com/images/G/github-logo-5F384D0265-seeklogo.com.png" alt="GitHub" width="80" height="80" style="border-radius: 25%;">
</a>

<br>
<br>

Kami menantikan kabar dari Anda dan mendukung perjalanan Anda dalam pembelajaran mesin!
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
