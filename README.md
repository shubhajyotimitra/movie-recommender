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

