import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

# Load dataset
def load_data():
    books = pd.read_csv('books.csv', on_bad_lines='skip', delimiter=None, engine="python")
    return books

# Fetch book cover image
def get_book_image(isbn):
    if pd.isna(isbn):
        return "https://via.placeholder.com/150?text=No+Cover"
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"

# Search for free book PDFs from PDFDrive
def get_pdfdrive_link(book_name):
    search_url = f"https://www.pdfdrive.com/search?q={book_name.replace(' ', '+')}"
    response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        book_links = soup.find_all('a', class_='ai-search')
        if book_links:
            return "https://www.pdfdrive.com" + book_links[0]['href']
    return None

# Book recommender
def recommend_books(book_name, books, tfidf_matrix):
    book_name = book_name.lower()
    matches = books[books['title'].str.lower().str.contains(book_name, na=False)]
    if matches.empty:
        return []
    idx = matches.index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    scores = list(enumerate(cosine_sim[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    book_indices = [i[0] for i in scores]
    return books.iloc[book_indices]

# Streamlit app
st.title("Book Recommender System")
books = load_data()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['title'].fillna(''))

# User input
book_name = st.text_input("Enter a book name, keyword, or sentence:")
if st.button("Search"):
    if book_name:
        st.subheader("Book Details")
        book_info = books[books['title'].str.lower().str.contains(book_name.lower(), na=False)]
        if not book_info.empty:
            book_info = book_info.iloc[0]
            st.image(get_book_image(book_info['isbn']), width=150)
            st.write(f"**Title:** {book_info['title']}")
            st.write(f"**Author:** {book_info['authors']}")
            st.write(f"**Average Rating:** {book_info['average_rating']}")
            st.write(f"**ISBN:** {book_info['isbn']}")
            st.write(f"**Published Date:** {book_info['publication_date']}")
            st.write(f"**Publisher:** {book_info['publisher']}")
            
            # Provide Free PDF Link
            pdf_link = get_pdfdrive_link(book_info['title'])
            if pdf_link:
                st.markdown(f"[Download Free PDF]({pdf_link})")
            else:
                st.write("No free PDF found.")
        else:
            st.write("No details found for this book.")
        
        st.subheader("Recommended Books")
        recommendations = recommend_books(book_name, books, tfidf_matrix)
        if recommendations.empty:
            st.write("No recommendations found.")
        else:
            cols = st.columns(2)
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                with cols[idx % 2]:
                    st.image(get_book_image(row['isbn']), width=150)
                    st.subheader(row['title'])
                    st.write(f"**Author:** {row['authors']}")
                    st.write(f"**Average Rating:** {row['average_rating']}")
                    st.write(f"**Published Date:** {row['publication_date']}")
                    st.write(f"**Publisher:** {row['publisher']}")
                    pdf_link = get_pdfdrive_link(row['title'])
                    if pdf_link:
                        st.markdown(f"[Download Free PDF]({pdf_link})")
                    else:
                        st.write("No free PDF available.")


    