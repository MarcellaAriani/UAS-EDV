import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =========================
# CONFIG PAGE
# =========================

st.set_page_config(
    page_title="Analisis Sentimen Jenius Bank",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("jenius_sentiment.csv")

# =========================
# HEADER
# =========================

st.title("Analisis Sentimen Jenius Bank")
st.markdown("### Eksplorasi dan Visualisasi Data")
st.markdown("---")

# =========================
# SIDEBAR
# =========================

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Home",
        "📊 Data Mentah",
        "🧹 Data Bersih",
        "☁️ WordCloud",
        "😊 Sentimen",
        "📈 Visualisasi"
    ]
)

# =========================
# HOME
# =========================

if menu == "🏠 Home":

    st.header("Selamat Datang")

    st.markdown("""
    ### 👩‍🎓 Identitas Mahasiswa

    **Nama :** Marcella Ariani  
    **NIM :** 20254920003  
    **Program Studi :** Statistika  
    **Mata Kuliah :** Eksplorasi dan Visualisasi Data  

    ---

    ### 📱 Analisis Sentimen Jenius Bank

    Aplikasi ini digunakan untuk melakukan analisis sentimen terhadap ulasan pengguna aplikasi Jenius Bank yang diambil dari Google Play Store.

    ### 🔎 Tahapan Analisis

    - Scraping Data
    - Preprocessing
    - WordCloud
    - Analisis Sentimen
    - Visualisasi Hasil
    """)

# =========================
# DATA MENTAH
# =========================

elif menu == "📊 Data Mentah":

    st.header("📊 Data Mentah")

    st.write(
        "Data hasil scraping dari Google Play Store sebelum dilakukan preprocessing."
    )

    st.dataframe(
        df[['reviewId', 'content', 'score']]
    )
    
    st.markdown("""
    ### 📝 Penjelasan

    Data mentah merupakan hasil scraping sebanyak 1000 ulasan pengguna aplikasi Jenius Bank dari Google Play Store.
    Data yang diperoleh terdiri dari reviewId sebagai identitas ulasan, content sebagai isi ulasan pengguna, dan score sebagai rating yang diberikan pengguna terhadap aplikasi.
    """)

# =========================
# DATA BERSIH
# =========================

elif menu == "🧹 Data Bersih":

    st.header("🧹 Data Setelah Preprocessing")

    st.write(
        "Data yang telah dibersihkan melalui proses preprocessing."
    )

    st.dataframe(
        df[['content', 'cleaned_text']]
    )
    st.markdown("""
    ### 📝 Penjelasan

    Tahap preprocessing dilakukan untuk membersihkan teks dari karakter yang tidak diperlukan seperti URL, angka, emoji, tanda baca, hashtag, dan stopwords.
    Tujuannya adalah agar data lebih terstruktur dan mudah dianalisis oleh model machine learning.
    """)
    
# =========================
# WORDCLOUD
# =========================

elif menu == "☁️ WordCloud":

    st.header("☁️ WordCloud")

    text = " ".join(
        df["cleaned_text"].fillna("").astype(str)
    )

    if text.strip():

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="#FFF5F7",
            colormap="RdPu"
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig)

        st.markdown("""
        ### 📝 Penjelasan       

        WordCloud menampilkan kata-kata yang paling sering muncul pada ulasan pengguna Jenius Bank.
        Semakin besar ukuran kata, maka semakin sering kata tersebut muncul dalam kumpulan ulasan.
        Visualisasi ini membantu mengidentifikasi topik utama yang sering dibahas oleh pengguna aplikasi.
        """) 
    else:
        st.warning("Data teks kosong.")

# =========================
# SENTIMENT
# =========================

elif menu == "😊 Sentimen":

    st.header("😊 Hasil Analisis Sentimen")

    st.dataframe(
        df[
            [
                'content',
                'score',
                'sentiment_label'
            ]
        ]
    )

    st.markdown("""
    ### 📝 Penjelasan   

    Analisis sentimen dilakukan menggunakan metode TF-IDF dan algoritma Logistic Regression.
    Setiap ulasan diklasifikasikan ke dalam dua kategori, yaitu sentimen positif dan sentimen negatif berdasarkan pola kata yang terdapat dalam ulasan.
    
    """)
# =========================
# VISUALISASI
# =========================

elif menu == "📈 Visualisasi":

    st.header("📈 Distribusi Sentimen")

    sentiment_counts = (
        df['sentiment_label']
        .value_counts()
    )

    # BAR CHART
    fig1, ax1 = plt.subplots(figsize=(8,5))

    ax1.bar(
        sentiment_counts.index,
        sentiment_counts.values,
        color=['#F8C8DC', '#E7D3FA']
    )

    ax1.set_title("Distribusi Sentimen")

    st.pyplot(fig1)

    st.markdown("""
    ### 📝 Penjelasan Bar Chart

    Grafik batang menunjukkan jumlah ulasan positif dan negatif yang berhasil diklasifikasikan oleh model.
    Grafik ini membantu melihat kategori sentimen yang paling dominan pada ulasan pengguna Jenius Bank.
    """)


    # PIE CHART
    fig2, ax2 = plt.subplots(figsize=(6,6))

    ax2.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=['#F8C8DC', '#E7D3FA'],
        startangle=90
    )

    ax2.set_title("Persentase Sentimen")

    st.pyplot(fig2)

    positive = sentiment_counts.get('Positive', 0)
    negative = sentiment_counts.get('Negative', 0)

    st.markdown(f"""
    ### 📝 Penjelasan Pie Chart

    Berdasarkan hasil analisis sentimen, diperoleh **{positive} ulasan positif** dan **{negative} ulasan negatif**.
    Hal ini menunjukkan bahwa mayoritas pengguna memberikan tanggapan {'positif' if positive > negative else 'negatif'} terhadap aplikasi Jenius Bank.
    """)