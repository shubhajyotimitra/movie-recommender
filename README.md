# 🎬 Movie Recommendation Chatbot (Hybrid AI System)

## 🚀 Overview

This project is a **hybrid movie recommendation system** that combines:

* **Machine Learning (TF-IDF + Cosine Similarity)** for content-based recommendations
* **Rule-based / LLM-inspired logic** for understanding user preferences
* **Interactive UI** built using Streamlit
* **External API integration** (TMDB) to display movie posters

The system allows users to describe what kind of movies they like in natural language and returns relevant recommendations.

---

## 🧠 Problem Statement

Traditional recommendation systems struggle to:

* Understand **natural language input**
* Handle **user constraints** (e.g., “not horror”)
* Provide an **interactive experience**

This project solves that by combining:

* ML for similarity
* Logic/LLM-style parsing for intent

---

## 🏗️ System Architecture

User Input
⬇
Preference Extraction (Include / Exclude)
⬇
TF-IDF Vectorization
⬇
Cosine Similarity Matching
⬇
Filtering (Constraints Applied)
⬇
Final Recommendations + Posters

---

## ⚙️ Tech Stack

* Python
* Pandas
* Scikit-learn
* Streamlit
* TMDB API
* Requests

---

## 🔍 Key Features

✅ Content-based recommendation system
✅ Hybrid AI (ML + logic-based understanding)
✅ Handles user constraints (e.g., “not action”)
✅ Interactive UI
✅ Movie posters using TMDB API

---

## 🧪 Example Queries

* “funny action movie in space”
* “romantic movie but not action”
* “crime thriller”

---

## 📸 Screenshots

(Add screenshots of your app here)

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add TMDB API key

In `app.py`, replace:

```python
api_key = "YOUR_TMDB_API_KEY"
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
movie-recommender/
│
├── app.py
├── requirements.txt
├── data/
│   └── tmdb_5000_movies.csv
```

---

## ⚠️ Limitations

* Uses **rule-based preference extraction** (not full LLM)
* No user history / personalization
* Limited understanding of complex language

---

## 🚀 Future Improvements

* Integrate real LLM (OpenAI / Hugging Face)
* Add user feedback (like/dislike)
* Implement collaborative filtering
* Deploy on cloud for public access

---

## 💡 Key Learnings

* Built an end-to-end ML system
* Learned hybrid system design (ML + AI logic)
* Integrated APIs into ML pipeline
* Developed interactive UI

---

## 📌 Resume Highlight

> Developed a hybrid movie recommendation chatbot using TF-IDF and cosine similarity, integrated with intelligent preference parsing and deployed via Streamlit with TMDB API-based UI enhancements.

---

## 🙌 Acknowledgements

* TMDB Dataset
* Scikit-learn
* Streamlit
