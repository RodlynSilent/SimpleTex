import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def extract_keywords(text):
    # Create a TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Fit and transform the text with the vectorizer
    tfidf_matrix = tfidf.fit_transform([text])

    # Get feature names to use as keywords
    feature_names = tfidf.get_feature_names_out()

    # Extract scores
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # Create a dictionary of keywords and their corresponding TF-IDF scores
    keyword_scores = dict(zip(feature_names, tfidf_scores))

    # Sort keywords by scores in descending order
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top 10 keywords
    top_keywords = sorted_keywords[:10]

    return top_keywords

# Set up the Streamlit interface
st.title('Simple Keyword Extractor')

# User text input
user_input = st.text_area("Enter text here:", height=300)
if user_input:
    # Extract keywords
    keywords = extract_keywords(user_input)

    # Display the keywords
    st.subheader('Extracted Keywords:')
    for word, score in keywords:
        st.write(f"{word} (Score: {score:.2f})")

    # Optional: Display a word cloud
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wordcloud = WordCloud(width=800, height=400, background_color ='white').generate_from_frequencies(dict(keywords))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    except ImportError:
        st.error("WordCloud module is not installed. Run 'pip install wordcloud' to enable this feature.")

# Instructions and documentation
st.sidebar.header("Instructions")
st.sidebar.write("1. Paste or type your text into the text area above.")
st.sidebar.write("2. Press Enter or click outside the text box to process the text.")
st.sidebar.write("3. View the extracted keywords and their importance scores below the text box.")
st.sidebar.write("4. If installed, view the optional word cloud visualizing keyword importance.")
