import pandas as pd
import streamlit as st
import cleantext
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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pytz

nltk.download('stopwords')
nltk.download('vader_lexicon')

check_session_timeout()

# Get Firestore client
db = get_firestore_client()

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to perform sentiment analysis using VADER
def perform_sentiment_analysis(text):
    # Clean the text
    cleaned_text = cleantext.clean(text, clean_all=False, extra_spaces=True,
                                   stopwords=True, lowercase=True,
                                   numbers=True, punct=True)
    sentiment_scores = sid.polarity_scores(cleaned_text)
    polarity = round(sentiment_scores['compound'], 2)
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

def generate_pdf_report(analyzed_df, overall_score, sentiment_counts, wordcloud_buffer, sentiment_score_fig, sentiment_pie_fig, topic_words_fig, lda_model, feature_names):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter
    
    # Function to add a new page if the current page is full
    def check_y_position(y_position, min_y, content_height):
        if y_position - content_height < min_y:
            pdf.showPage()  # Add new page
            return page_height - 50
        return y_position

    # Initialize y_position
    y_position = page_height - 50

    # Add sentiment score bar chart
    img_buffer = BytesIO()
    sentiment_score_fig.set_size_inches(5, 2.5)  # Resize the figure
    sentiment_score_fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    sentiment_score_img = ImageReader(img_buffer)
    y_position = check_y_position(y_position, 50, 200)  # Check if the content fits
    pdf.drawImage(sentiment_score_img, 100, y_position - 200, width=400, height=200)
    y_position -= 250  # Adjust y_position after adding the image

    # Add sentiment pie chart
    img_buffer = BytesIO()
    sentiment_pie_fig.set_size_inches(5, 3)  # Resize the figure
    sentiment_pie_fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    sentiment_pie_img = ImageReader(img_buffer)
    y_position = check_y_position(y_position, 50, 200)  # Check if the content fits
    pdf.drawImage(sentiment_pie_img, 100, y_position - 200, width=400, height=200)
    y_position -= 250  # Adjust y_position after adding the image

    # Add word cloud
    y_position = check_y_position(y_position, 50, 300)  # Check if the content fits
    wordcloud_img = ImageReader(wordcloud_buffer)
    pdf.drawImage(wordcloud_img, 100, y_position - 300, width=400, height=300)
    y_position -= 350  # Adjust y_position after adding the image

    # Add top words bar chart for each topic
    img_buffer = BytesIO()
    topic_words_fig.set_size_inches(5, 3)  # Resize the figure
    topic_words_fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    topic_words_img = ImageReader(img_buffer)
    y_position = check_y_position(y_position, 50, 200)  # Check if the content fits
    pdf.drawImage(topic_words_img, 100, y_position - 200, width=400, height=200)
    y_position -= 250  # Adjust y_position after adding the image

    # Add topics from LDA
    y_position = check_y_position(y_position, 50, 20)  # Check if the content fits
    pdf.drawString(100, y_position, "Topics Identified:")
    y_position -= 20
    for topic_idx, topic in enumerate(lda_model.components_):
        y_position = check_y_position(y_position, 50, 20)  # Check if the content fits
        topic_words = " ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
        pdf.drawString(100, y_position, f"Topic {topic_idx+1}: {topic_words}")
        y_position -= 20

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

# Function to fetch and display upload history with delete option
def view_upload_history(username):
    try:
        datasets_ref = db.collection("datasets").where("username", "==", username).stream()

        history = []
        for dataset in datasets_ref:
            data = dataset.to_dict()
            file_name = data.get("file_name", "Unknown File")  # Use a default value if 'file_name' is missing
            timestamp = data.get("timestamp", "Unknown Timestamp")  # Use a default value if 'timestamp' is missing
            
            if isinstance(timestamp, datetime):
                # Convert timestamp to the desired time zone (e.g., UTC+8)
                target_time_zone = pytz.timezone("Asia/Shanghai")
                local_timestamp = timestamp.astimezone(target_time_zone)
                formatted_timestamp = local_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_timestamp = timestamp

            # Append the file name, timestamp, and document ID to the history list
            history.append({"File Name": file_name, "Upload Time": formatted_timestamp, "doc_id": dataset.id})

        # Sort the history list by 'Upload Time' in descending order
        history = sorted(history, key=lambda x: x["Upload Time"], reverse=True)

        # Display the history with delete buttons
        st.write("Upload History:")
        for entry in history:
            col1, col2, col3 = st.columns([4, 4, 1])
            col1.write(entry["File Name"])
            col2.write(entry["Upload Time"])
            if col3.button("Delete", key=f"delete_{entry['doc_id']}"):
                st.session_state.confirm_delete = entry["doc_id"]
                st.session_state.file_name_to_delete = entry["File Name"]
                st.experimental_rerun()  # Rerun the app to refresh the state

        # Display confirmation modal if a delete action is triggered
        if st.session_state.get("confirm_delete"):
            st.write(f"Are you sure you want to delete the file '{st.session_state.file_name_to_delete}'?")
            confirm_col, cancel_col = st.columns([1, 1])
            if confirm_col.button("Confirm"):
                delete_history_entry(st.session_state.confirm_delete)
                st.session_state.confirm_delete = None
                st.session_state.file_name_to_delete = None
                st.experimental_rerun()
            if cancel_col.button("Cancel"):
                st.session_state.confirm_delete = None
                st.session_state.file_name_to_delete = None
                st.experimental_rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Function to delete a history entry from Firestore
def delete_history_entry(doc_id):
    try:
        db.collection("datasets").document(doc_id).delete()
        st.success("Deleted successfully.")
    except Exception as e:
        st.error(f"An error occurred while deleting: {e}")

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

    return lda_model, feature_names

def comments_analyser():
    st.title("Comments Analyser")

    selected = st.selectbox("Select Section", ["Upload Data", "View History"])

    if selected == "Upload Data":
        st.header("Upload Data")
        upl = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])

        if upl:
            df = pd.read_csv(upl) if upl.name.endswith('.csv') else pd.read_excel(upl, engine='openpyxl')
            st.write("First ten rows of Original Data:")
            st.write(df.head(10))

            # Perform phrase extraction and sentiment analysis
            analyzed_df = analyze_dataframe(df)

            st.write("First ten rows of Analyzed Data:")
            st.write(analyzed_df.head(10))
            
            # Save dataset to Firestore
            file_name = upl.name
            dataset_content = df.to_csv(index=False)
            save_dataset_to_firestore(st.session_state.username, dataset_content, file_name)

            st.write("Dataset is analyzed and saved successfully!")

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

            # Perform topic modeling
            lda_model, feature_names = perform_topic_modeling(cleaned_column)
            for topic_idx, topic in enumerate(lda_model.components_):
                st.write(f"Topic {topic_idx+1}:")
                st.write(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

            # Bar chart for top words in each topic
            topic_word_data = {}
            for topic_idx, topic in enumerate(lda_model.components_):
                topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                topic_word_data[f'Topic {topic_idx+1}'] = topic_words

            selected_topic = st.sidebar.selectbox("Select Topic", list(topic_word_data.keys()))

            if selected_topic:
                fig5, ax5 = plt.subplots()
                words = topic_word_data[selected_topic]
                counts = [lda_model.components_[list(topic_word_data.keys()).index(selected_topic)][feature_names.tolist().index(word)] for word in words]
                ax5.barh(words, counts, color='skyblue')
                ax5.set_xlabel('Frequency')
                ax5.set_title(f'Top Words in {selected_topic}')
                ax5.invert_yaxis()
                st.pyplot(fig5)
            
            # Generate the PDF report
            pdf_buffer = generate_pdf_report(analyzed_df, overall_score, sentiment_counts, wordcloud_buffer, fig1, fig2, fig5, lda_model, feature_names)

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
