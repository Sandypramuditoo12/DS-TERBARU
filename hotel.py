import streamlit as st
import pickle
import numpy as np
from xgboost import XGBClassifier

# Fungsi untuk memuat model
def load_model(file_path):
    with open(file_path, 'rb') as file:
        classifier = pickle.load(file)
    return classifier

st.title("Cek Hotel Berdasarkan Rating dan Jumlah Review")

# Membuat input form
input0 = st.text_input("Nama Hotel", placeholder="Masukkan nama hotel")
input1 = st.text_input("Rating", placeholder="Masukkan rating (angka)")
input2 = st.text_input("Jumlah Review", placeholder="Masukkan jumlah review (angka)")

# Mapping prediksi ke pesan
def interpret_prediction(prediction_value):
    messages = {
        0: "Adalah Hotel Terkategori Exceptional",
        1: "Adalah Hotel Terkategori Superb",
        2: "Adalah Hotel Terkategori Very Good",
        3: "Adalah Hotel Terkategori Good",
        4: "Adalah Hotel Terkategori Fabulous"
    }
    return messages.get(prediction_value, "Hasil prediksi tidak valid.")

# Tombol untuk submit data
if st.button("Cek"):
    try:
        # Validasi input - memastikan semua input diisi
        if not (input0 and input1 and input2):
            st.error("Harap isi semua input!")
        else:
            # Konversi input ke format numerik
            try:
                input_data = np.array([float(input1), float(input2)]).reshape(1, -1)
            except ValueError:
                st.error("Rating dan Jumlah Review harus berupa angka!")
                st.stop()

            # Memuat model dari file
            model_path = 'hotel.pkl'  # Lokasi file model
            classifier = load_model(model_path)

            # Melakukan prediksi
            prediction = classifier.predict(input_data)

            # Menafsirkan hasil prediksi
            result_message = interpret_prediction(prediction[0])

            # Menampilkan hasil prediksi
            st.success(f"{input0} {result_message}")
    except FileNotFoundError:
        st.error("File model 'hotel.pkl' tidak ditemukan. Pastikan file tersedia di direktori.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
