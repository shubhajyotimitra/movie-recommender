import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Title
st.title("🎬 Movie Recommendation Chatbot")
st.markdown("### 🤖 AI-powered Movie Recommender")

# Load dataset
df = pd.read_csv("data/tmdb_5000_movies.csv")

# Preprocess
df = df[['id', 'title', 'genres', 'overview']]
df.dropna(inplace=True)

def extract_genres(text):
    genres = []
    for i in ast.literal_eval(text):
        genres.append(i['name'])
    return " ".join(genres)

df['genres'] = df['genres'].apply(extract_genres)
df['tags'] = df['overview'] + " " + df['genres']

# Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(df['tags']).toarray()

# Preference extraction (your hybrid logic)
def extract_preferences(user_input):
    user_input = user_input.lower()
    
    include = []
    exclude = []
    
    genres = ["action", "comedy", "romance", "thriller", "crime", "horror", "sci-fi"]
    
    for g in genres:
        if g in user_input:
            if "not " + g in user_input or "no " + g in user_input:
                exclude.append(g)
            else:
                include.append(g)
    
    return {"include": include, "exclude": exclude}

def fetch_poster(movie_id):
    api_key = "fc0a32c31ed208cb69e183ec5c8bc960"
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    
    data = requests.get(url).json()
    
    poster_path = data.get('poster_path')
    
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        return None

# Recommendation function
def recommend_hybrid(user_input):
    
    prefs = extract_preferences(user_input)
    include_text = " ".join(prefs['include'])
    
    user_vector = tfidf.transform([include_text]).toarray()
    scores = cosine_similarity(user_vector, vectors)
    
    movie_list = sorted(list(enumerate(scores[0])), reverse=True, key=lambda x: x[1])[1:20]
    
    final_movies = []
    
    for i in movie_list:
        movie_id = df.iloc[i[0]]['id']
        title = df.iloc[i[0]].title
        tags = df.iloc[i[0]].tags.lower()
        
        if any(word in tags for word in prefs['exclude']):
            continue
        
        final_movies.append((title, movie_id))
        
        if len(final_movies) == 5:
            break
    
    return final_movies

# User input
user_input = st.text_input("Describe what kind of movies you like:")

if user_input:
    recommendations = recommend_hybrid(user_input)
    
    st.subheader("🎯 Recommended Movies:")
    
    cols = st.columns(5)
    
    for idx, (title, movie_id) in enumerate(recommendations):
        poster = fetch_poster(movie_id)
        
        with cols[idx]:
            if poster:
                st.image(poster)
            st.write(title)