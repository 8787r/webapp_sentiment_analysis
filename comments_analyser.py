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
from google.cloud import firestore
from auth import get_firestore_client, check_session_timeout
from datetime import datetime, timedelta
from io import StringIO

nltk.download('stopwords')

check_session_timeout()

# Get Firestore client
db = get_firestore_client()

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

def generate_pdf_report(analyzed_df, overall_score, sentiment_counts, wordcloud_buffer, word_frequencies, sentiment_score_fig, sentiment_pie_fig, top_words_fig):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter
    
    # Overall Sentiment Score
    pdf.drawString(100, page_height - 50, f"Overall Sentiment Score: {overall_score:.2f}")
    
    # Percentage of Sentiments
    pdf.drawString(100, page_height - 70, f"Percentage of Positive Sentiment: {sentiment_counts.get('Positive', 0):.2f} %")
    pdf.drawString(100, page_height - 90, f"Percentage of Negative Sentiment: {sentiment_counts.get('Negative', 0):.2f} %")
    pdf.drawString(100, page_height - 110, f"Percentage of Neutral Sentiment: {sentiment_counts.get('Neutral', 0):.2f} %")
    
    # Word Cloud
    wordcloud_img = ImageReader(wordcloud_buffer)  # Convert BytesIO to ImageReader
    pdf.drawImage(wordcloud_img, 100, page_height - 500, width=400, height=300)

    # Top Ten Words with Highest Frequency
    pdf.drawString(100, page_height - 550, "Top Ten Words with Highest Frequency:")
    y_position = page_height - 570
    for word, frequency in word_frequencies.items():
        pdf.drawString(100, y_position, f"{word}: {frequency}")
        y_position -= 20

    # Add sentiment score bar chart
    img_buffer = BytesIO()
    sentiment_score_fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    sentiment_score_img = ImageReader(img_buffer)
    pdf.drawImage(sentiment_score_img, 100, page_height - 800, width=400, height=200)

    # Add sentiment pie chart
    img_buffer = BytesIO()
    sentiment_pie_fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    sentiment_pie_img = ImageReader(img_buffer)
    pdf.drawImage(sentiment_pie_img, 100, page_height - 1100, width=400, height=200)

    # Add top words bar chart
    img_buffer = BytesIO()
    top_words_fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    top_words_img = ImageReader(img_buffer)
    pdf.drawImage(top_words_img, 100, page_height - 1400, width=400, height=200)

    pdf.save()
    
    buffer.seek(0)
    return buffer

# Function to save dataset to Firestore in chunks
def save_dataset_to_firestore(username, dataset_content, file_name, max_chunk_size=1048000):
    # Read the CSV content into a DataFrame
    df = pd.read_csv(StringIO(dataset_content))
    
    # Convert the DataFrame back to CSV format in chunks
    csv_chunks = []
    current_chunk = StringIO()
    total_size = 0

    for i, row in df.iterrows():
        row_csv = row.to_frame().T.to_csv(index=False, header=False)
        row_size = len(row_csv.encode('utf-8'))
        
        if total_size + row_size > max_chunk_size:
            csv_chunks.append(current_chunk.getvalue())
            current_chunk = StringIO()
            total_size = 0
        
        current_chunk.write(row_csv)
        total_size += row_size

    if total_size > 0:
        csv_chunks.append(current_chunk.getvalue())

    # Save each chunk to Firestore
    for idx, chunk in enumerate(csv_chunks):
        dataset_data = {
            "username": username,
            "timestamp": datetime.utcnow(),
            "chunk_id": idx,
            "dataset": chunk,
            "file_name": file_name
        }
        db.collection("datasets").add(dataset_data)

# Function to get datasets for a user from Firestore
def get_datasets_for_user(username):
    datasets_ref = db.collection("datasets").where("username", "==", username).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    datasets = [dataset.to_dict() for dataset in datasets_ref]
    return datasets

# Function to fetch and display upload history
def view_upload_history(username):
    try:
        datasets_ref = db.collection("datasets").where("username", "==", username).stream()
        
        history = []
        for dataset in datasets_ref:
            data = dataset.to_dict()
            file_name = data.get("file_name", "Unknown File")  # Use a default value if 'file_name' is missing
            timestamp = data.get("timestamp", "Unknown Timestamp")  # Use a default value if 'timestamp' is missing
            
            if isinstance(timestamp, datetime):
                formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_timestamp = timestamp

            # Append the file name and timestamp to the history list
            history.append({"File Name": file_name, "Upload Time": formatted_timestamp})

        # Sort the history list by 'Upload Time' in descending order
        history = sorted(history, key=lambda x: x["Upload Time"], reverse=True)

        # Convert the history list to a DataFrame
        history_df = pd.DataFrame(history)

        # Display the history in table format
        st.write("Upload History:")
        st.write(history_df.to_html(index=False), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

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
            st.write(f"Topic {topic_idx+1}:")
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
            
            # Save dataset to Firestore
            file_name = upl.name
            dataset_content = df.to_csv(index=False)
            save_dataset_to_firestore(st.session_state.username, dataset_content, file_name)

            st.write("Dataset saved successfully!")

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
            overall_score_percentage = round((overall_score + 1) / 2 * 100, 2)  # Scale to [0, 100] for visualization

            fig1, ax1 = plt.subplots(figsize=(8, 2))
            ax1.barh(['Overall Sentiment Score'], [100], color='lightgray')  # Full percentage bar
            ax1.barh(['Overall Sentiment Score'], [overall_score_percentage], color='blue')  # Score percentage bar
            ax1.set_xlabel('Percentage')
            ax1.set_title('Overall Sentiment Score')
            ax1.set_xlim(0, 100)
            ax1.invert_yaxis()  # Invert y-axis to have the bar extend from left to right
            ax1.text(overall_score_percentage + 2, 0, f'{overall_score_percentage}%', va='center')
            st.pyplot(fig1)

            # Percentage of Positive/Negative/Neutral Sentiment
            sentiment_counts = analyzed_df['Sentiment'].value_counts(normalize=True) * 100
            labels = sentiment_counts.index
            sizes = sentiment_counts.values
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax2.set_title('Percentage of Sentiment')
            st.pyplot(fig2)

            # Perform data cleaning on the text column
            cleaned_column = analyzed_df[analyzed_df.columns[0]].apply(lambda x: cleantext.clean(x, clean_all=False, extra_spaces=True,
                                                                                stopwords=True, lowercase=True,
                                                                                numbers=True, punct=True))

            # Concatenate cleaned text
            clean_text = ' '.join(cleaned_column)

            # Generate Word Cloud
            wordcloud = WordCloud().generate(clean_text)
            fig3, ax3 = plt.subplots()
            ax3.imshow(wordcloud, interpolation='bilinear')
            ax3.axis("off")
            st.pyplot(fig3)
            wordcloud_buffer = generate_wordcloud(clean_text)

            # Top Ten Words with Highest Frequency
            word_frequencies = pd.Series(clean_text.split()).value_counts()[:10]
            words = word_frequencies.index.tolist()
            frequencies = word_frequencies.values.tolist()
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            bars = ax4.bar(words, frequencies, color='skyblue')
            ax4.set_xlabel('Words')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Top Ten Words with Highest Frequency')
            for bar, freq in zip(bars, frequencies):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(freq), ha='center', va='bottom')
            ax4.set_xticks(ax4.get_xticks())
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig4)

            # Perform topic modeling
            perform_topic_modeling(cleaned_column)

            # Generate the WordCloud
            wordcloud_buffer = generate_wordcloud(clean_text)

            # Assuming you have analyzed_df, overall_score, sentiment_counts, wordcloud_buffer, and word_frequencies
            pdf_buffer = generate_pdf_report(analyzed_df, overall_score, sentiment_counts, wordcloud_buffer, word_frequencies, fig1, fig2, fig4)

            
            st.download_button(
                label="Download Report as PDF",
                data=pdf_buffer,
                file_name="sentiment_analysis_report.pdf",
                mime="application/pdf",
            )

    elif selected == "View History":
        st.header("View History")
        username = st.session_state.username
        datasets = get_datasets_for_user(username)
        if datasets:
            # display_datasets(datasets)
            view_upload_history(st.session_state.username)
        else:
            st.write("No datasets found.")

if __name__ == "__main__":
    comments_analyser()
