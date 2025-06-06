import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ------------------ Spotify API Setup ------------------
CLIENT_ID = "2b3e64a704f144eb82fbe38e91ac511d"
CLIENT_SECRET = "980ecca21c6d45f6a3a07c4a1e96ffdd"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

def get_album_cover(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")
    if results and results["tracks"]["items"]:
        return results["tracks"]["items"][0]["album"]["images"][0]["url"]
    return "https://i.postimg.cc/0QNxYz4V/social.png"

# ------------------ Data Preparation ------------------
df = pd.read_csv("ex.csv")
df.fillna('', inplace=True)
df = df.rename(columns={
    'Song-Name': 'song',
    'Singer/Artists': 'artist',
    'Genre': 'genre',
    'Album/Movie': 'album',
    'User-Rating': 'rating'
})

df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df['combined'] = df['song'] + ' ' + df['artist'] + ' ' + df['genre'] + ' ' + df['album'] + ' ' + df['rating']

# ------------------ TF-IDF Vectorization ------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ------------------ Recommendation Function ------------------
def recommend(song_title):
    song_title = song_title.lower()
    if song_title not in df['song'].values:
        return [], []

    idx = df[df['song'] == song_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    recommended_names = []
    recommended_covers = []

    for i in sim_scores:
        title = df.iloc[i[0]]['song'].title()
        artist = df.iloc[i[0]]['artist'].title()
        cover_url = get_album_cover(title, artist)
        recommended_names.append(title)
        recommended_covers.append(cover_url)

    return recommended_names, recommended_covers

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="🎶 Music Recommendation", layout="wide", page_icon="🎧")
st.header("🎧 Music Recommendation System")

song_list = df['song'].str.title().unique()
selected_song = st.selectbox("🎵 Choose a song you like", sorted(song_list))

if st.button("🔍 Show Recommendations"):
    names, covers = recommend(selected_song)
    if names:
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.text(names[i])
                st.image(covers[i])
    else:
        st.error("❌ Song not found in dataset.")

