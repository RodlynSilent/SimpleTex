import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image

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

# Load the logo image
logo_path = "simpletex_logo.png"
logo = Image.open(logo_path)

# Set up the Streamlit interface
st.image(logo, width=150)  # Resize the logo to a smaller width
st.title('Simple Keyword Extractor')

# Custom CSS to make content use full width
st.markdown("""
<style>
    .block-container {
        padding: 2rem;
        max-width: 100%;
    }
    .stContainer {
        width: 100%;
    }
    .css-1d391kg {
        background-color: #90e0ef;
    }
    .stApp {
        background-color: #ffb703;
    }
    .stTextArea textarea {
        background-color: #e0e0e0;  /* Background color */
        color: #333333;  /* Text color */
        font-size: 16px;  /* Font size */
        border: 2px solid #cccccc;  /* Border style */
        border-radius: 5px;  /* Rounded corners */
        padding: 10px;  /* Padding inside the text area */
    }        
</style>
""", unsafe_allow_html=True)

# Create two columns for layout of equal width
col1, col2 = st.columns(2)

# Left column for user input and instructions
with col1:
    user_input = st.text_area("Enter text here:", height=250)
    if st.button("Sumbit"): 
        st.write("You have submitted the text. Please wait for the results...")

    with st.expander("How to Use This Tool"):
        st.write("""
        - **Step 1:** Paste or type your text into the text area above.
        - **Step 2:** Press Enter or click outside the text box to process the text.
        - **Step 3:** View the extracted keywords and their importance scores below the text box.
        - **Step 4:** If installed, view the optional word cloud visualizing keyword importance.
        """)

# Right column for displaying results
with col2:
    if user_input:
        # Extract keywords
        keywords = extract_keywords(user_input)

        # Display the keywords
        st.subheader('Extracted Keywords:')
        col1, col2 = st.columns(2)
        half = len(keywords) // 2
        for i, (word, score) in enumerate(keywords):
            if i < half:
                col1.write(f"{word} (Score: {score:.2f})")
            else:
                col2.write(f"{word} (Score: {score:.2f})")

        # Optional: Display a word cloud
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(keywords))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        except ImportError:
            st.error("WordCloud module is not installed. Please run 'pip install wordcloud' to enable this feature. Thanks!")
