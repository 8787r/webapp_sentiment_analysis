import pandas as pd
import streamlit as st
import cleantext
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    # Clean the text
    cleaned_text = cleantext.clean(text, clean_all=False, extra_spaces=True,
                                   stopwords=True, lowercase=True,
                                   numbers=True, punct=True)
    blob = TextBlob(cleaned_text)
    polarity = round(blob.sentiment.polarity, 2)
    sentiment = analyze(polarity)
    return polarity, sentiment

def analyze(polarity):
    if polarity > 0.2:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Function to perform phrase extraction and sentiment analysis on a DataFrame
def analyze_dataframe(df):
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if column contains textual data
            df['Polarity'], df['Sentiment'] = zip(*df[column].apply(perform_sentiment_analysis))
    return df

def generate_wordcloud(clean_text):
    wordcloud = WordCloud().generate(clean_text)
    # Save WordCloud as an image
    img_buffer = BytesIO()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    return img_buffer

def generate_pdf_report(overall_score, sentiment_counts, wordcloud_buffer, word_frequencies):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter
    
    # Overall Sentiment Score
    pdf.drawString(100, page_height - 50, f"Overall Sentiment Score: {overall_score}")
    
    # Percentage of Sentiments
    pdf.drawString(100, page_height - 70, f"Percentage of Positive Sentiment: {sentiment_counts.get('Positive', 0)} %")
    pdf.drawString(100, page_height - 90, f"Percentage of Negative Sentiment: {sentiment_counts.get('Negative', 0)} %")
    pdf.drawString(100, page_height - 110, f"Percentage of Neutral Sentiment: {sentiment_counts.get('Neutral', 0)} %")
    
    # Word Cloud
    wordcloud_img = ImageReader(wordcloud_buffer)  # Convert BytesIO to ImageReader
    pdf.drawImage(wordcloud_img, 100, page_height - 300, width=400, height=300)

    # Top Ten Words with Highest Frequency
    pdf.drawString(100, page_height - 350, "Top Ten Words with Highest Frequency:")
    y_position = page_height - 370
    max_items_per_page = 30  # Adjust as needed
    current_page = 1
    start_index = 0

    for index, (word, frequency) in enumerate(word_frequencies.items()):
        if index % max_items_per_page == 0 and index > 0:
            # Start a new page after reaching the maximum items per page
            pdf.showPage()
            current_page += 1
            y_position = page_height - 50  # Adjust as needed
            pdf.drawString(100, y_position, f"Page {current_page}")

        pdf.drawString(100, y_position, f"{word}: {frequency}")
        y_position -= 20

    pdf.save()
    
    buffer.seek(0)
    return buffer

def comments_analyser():
    st.title("Comments Analyser")

    # Function to perform topic modeling
    def perform_topic_modeling(text_data):
        # Initialize CountVectorizer
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

        # Fit and transform your text data
        X = vectorizer.fit_transform(text_data)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Initialize LDA model
        num_topics = 5
        lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online', random_state=42)

        # Fit LDA to the transformed data
        lda_model.fit(X)

        # Display topics
        display_topics(lda_model, feature_names, num_top_words=10)

    def display_topics(model, feature_names, num_top_words):
        for topic_idx, topic in enumerate(model.components_):
            st.write(f"Topic {topic_idx}:")
            st.write(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

    selected = st.selectbox("Select Option", ["Upload Data", "View History"])

    if selected == "Upload Data":
        st.header("Upload Data")
        upl = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])

        if upl:
            df = pd.read_csv(upl) if upl.name.endswith('.csv') else pd.read_excel(upl, engine='openpyxl')
            st.write("Original Data:")
            st.write(df.head(10))

            # Perform phrase extraction and sentiment analysis
            analyzed_df = analyze_dataframe(df)

            st.write("Analyzed Data:")
            st.write(analyzed_df.head(10))
            
            # Download analyzed data as CSV
            @st.cache_data
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df(analyzed_df)

            st.download_button(
                label="Download Analyzed Data as CSV",
                data=csv,
                file_name='analyzed_data.csv',
                mime='text/csv',
            )
            
            # Overall Sentiment Score
            overall_score = analyzed_df['Polarity'].mean()
            overall_score_percentage = round(overall_score * 100, 2)
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Overall Sentiment Score'], [100], color='lightgray')  # Full percentage bar
            ax.barh(['Overall Sentiment Score'], [overall_score_percentage], color='blue')  # Score percentage bar
            ax.set_xlabel('Percentage')
            ax.set_title('Overall Sentiment Score')
            ax.set_xlim(0, 100)
            ax.invert_yaxis()  # Invert y-axis to have the bar extend from left to right
            ax.text(overall_score_percentage + 2, 0, f'{overall_score_percentage}%', va='center')
            ax.legend()
            st.pyplot(fig)

            # Percentage of Positive/Negative/Neutral Sentiment
            sentiment_counts = analyzed_df['Sentiment'].value_counts(normalize=True) * 100
            labels = sentiment_counts.index
            sizes = sentiment_counts.values
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title('Percentage of Sentiment')
            st.pyplot(fig)

            # Perform data cleaning on the text column
            cleaned_column = analyzed_df[analyzed_df.columns[0]].apply(lambda x: cleantext.clean(x, clean_all=False, extra_spaces=True,
                                                                                stopwords=True, lowercase=True,
                                                                                numbers=True, punct=True))

            # Concatenate cleaned text
            clean_text = ' '.join(cleaned_column)

            # Generate Word Cloud
            wordcloud = WordCloud().generate(clean_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            wordcloud_buffer = generate_wordcloud(clean_text)

            # Top Ten Words with Highest Frequency
            word_frequencies = pd.Series(clean_text.split()).value_counts()[:10]
            words = word_frequencies.index.tolist()
            frequencies = word_frequencies.values.tolist()
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(words, frequencies, color='skyblue')
            ax.set_xlabel('Words')
            ax.set_ylabel('Frequency')
            ax.set_title('Top Ten Words with Highest Frequency')
            for bar, freq in zip(bars, frequencies):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(freq), ha='center', va='bottom')
            ax.set_xticks(ax.get_xticks())
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

            # Perform topic modeling
            perform_topic_modeling(cleaned_column)

            # Generate the WordCloud
            wordcloud_buffer = generate_wordcloud(clean_text)

            pdf_buffer = generate_pdf_report(overall_score, sentiment_counts, wordcloud_buffer, word_frequencies)
            
            st.download_button(
                label="Download Report as PDF",
                data=pdf_buffer,
                file_name="sentiment_analysis_report.pdf",
                mime="application/pdf",
            )

    elif selected == "View History":
        st.header("View History")
        st.write("Coming soon...")
