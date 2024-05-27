import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import base64

st.set_page_config(page_title="SimpleTex: Your Simple Keyword Extractor")

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

def load_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Load the logo image
logo_path = "simpletex_logo.png"
logo_base64 = load_image(logo_path)

# Custom CSS to style the app
st.markdown(f"""
    <style>
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            background-color: #b6a081;
            border-radius: 10px;
            border: 1px solid #F0C775;
            background: #B6A081;
            backdrop-filter: blur(2px);
            width: 80%;
            margin: auto;
        }}
        .stApp {{
            background-color: #284867;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .stTextArea textarea {{
            background-color: #f5f5dc;
            color: #333333;
            font-size: 16px;
            border: 2px solid #cccccc;
            border-radius: 10px;
            padding: 10px;
            transition: border-color 0.3s;
        }}
        .stTextArea textarea:focus {{
            border-color: #F0C775;
        }}
        .stButton button {{
            background-color: #284867;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px 20px;
            transition: background-color 0.3s;
        }}
        .stButton button:active {{
            background-color: #f0c775;
        }}
        .css-1aumxhk, .css-1v0mbdj, .css-1d391kg {{
            margin-bottom: 20px;
        }}
        .center-logo {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
        }}
        .logo-img {{
            max-width: 200px;
        }}
        h1.title {{
            margin-top: -30px;
            text-align: center;
            color: #4a4a4a;
        }}
        .expander-header {{
            color: #4a4a4a !important;
        }}
        .expander-content {{
            display: none; /* Hide expander content by default */
        }}
    </style>
    <div class="center-logo">
        <img class="logo-img" src="data:image/png;base64,{logo_base64}" />
    </div>
    <h1 class="title">Simple Keyword Extractor</h1>
""", unsafe_allow_html=True)

# Create two columns for layout of equal width
col1, col2 = st.columns(2)

# Left column for user input and instructions
with col1:
    user_input = st.text_area("Enter text here", height=250)
    submit_button = st.button("Submit")
    if submit_button and user_input.strip():
        st.write("You have submitted the text. Please wait for the results...")

    with st.expander("How to Use This Tool", expanded=False):
        st.markdown("""
        - **Step 1:** Paste or type your text into the text area above.
        - **Step 2:** Press Enter or click outside the text box to process the text.
        - **Step 3:** View the extracted keywords and their importance scores below the text box.
        - **Step 4:** If installed, view the optional word cloud visualizing keyword importance.
        """, unsafe_allow_html=True)

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
