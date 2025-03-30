# Book Recommender System

# # Overview

The Book Recommender System is a Streamlit-based web application that allows users to search for books, get recommendations based on their input, and access free PDFs of books from PDFDrive. It uses TF-IDF vectorization and cosine similarity for book recommendations.

# # Features

Search for books by title or keyword.

Display book details including cover image, author, rating, ISBN, and publisher.

Fetch free book PDFs from PDFDrive (if available).

Provide personalized book recommendations.

# # Installation

Prerequisites

Ensure you have Python 3.x installed along with the following dependencies:

pip install streamlit pandas numpy scikit-learn requests beautifulsoup4

# # Running the App

Clone this repository:

git clone https://github.com/your-username/book-recommender.git

Navigate to the project directory:

cd book-recommender

Run the Streamlit app:

streamlit run app.py

# # Usage

Enter a book name or keyword in the search box.

Click the Search button to fetch book details.

Click the Download Free PDF link (if available) to get the book.

View recommended books based on your search.

# # Technologies Used

Python

Streamlit (for web UI)

Pandas & NumPy (for data handling)

Scikit-learn (for TF-IDF and similarity calculation)

Requests & BeautifulSoup (for fetching book PDFs from PDFDrive)

# # Dataset

The application uses a book dataset (books.csv) that contains book details such as title, author, rating, ISBN, publication date, and publisher.

Contributing

Feel free to contribute by creating a pull request or opening an issue for feature requests or bug reports.

License

This project is open-source and available under the MIT License.
