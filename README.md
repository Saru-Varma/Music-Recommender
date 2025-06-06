# 🎧 Music Recommendation System

A content-based Music Recommendation System built with *Python, **Streamlit, and **Spotify API*.  
It suggests similar songs based on genre, artist, mood, tags, and lyrics using TF-IDF and cosine similarity.

---

## 🔍 Features

- Recommend top 5 similar songs based on user input
- Album cover art fetched using the *Spotify API*
- Interactive and responsive *Streamlit web interface*
- Text-based similarity using *TF-IDF Vectorization*

---

## 🧠 How It Works

- The dataset (ex.csv) contains song metadata: title, genre, artist, mood, tags, lyrics.
- A combined feature is created and vectorized using TfidfVectorizer.
- Cosine similarity measures song similarity.
- Spotify API is used to fetch album cover images in real-time.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/music-recommender.git
cd music-recommender
