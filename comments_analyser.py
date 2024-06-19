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
from auth import get_firestore_client, check_session_timeout, get_storage_bucket, storage
from datetime import datetime, timedelta
from io import StringIO
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pytz
import uuid
import pyrebase
from bertopic import BERTopic
# from fpdf import FPDF
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
import pyLDAvis
# import pyLDAvis.sklearn
import streamlit.components.v1 as components
import os

nltk.download('stopwords')
nltk.download('vader_lexicon')

check_session_timeout()

# Get Firestore client
db = get_firestore_client()

# Get Firebase Storage bucket
bucket = get_storage_bucket()

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

# def generate_wordcloud(clean_text):
#     wordcloud = WordCloud().generate(clean_text)
#     # Save WordCloud as an image
#     img_buffer = BytesIO()
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.savefig(img_buffer, format='png')
#     img_buffer.seek(0)
#     return img_buffer

def upload_pdf_to_storage(pdf_buffer, file_name):
    bucket = get_storage_bucket()
    blob = bucket.blob(file_name)
    pdf_buffer.seek(0)  # Reset buffer position to the beginning
    blob.upload_from_file(pdf_buffer, content_type='application/pdf')
    blob.make_public()
    return blob.public_url

def generate_pdf_report(analyzed_df, overall_score, sentiment_counts, wordcloud_pos_buffer, wordcloud_neg_buffer, sentiment_score_fig, sentiment_pie_fig, topic_model, feature_names, is_lda):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter

    def check_y_position(y_position, min_y, content_height):
        if y_position - content_height < min_y:
            pdf.showPage()
            return page_height - 50
        return y_position

    y_position = page_height - 50

    # Add sentiment score bar chart
    img_buffer = BytesIO()
    sentiment_score_fig.set_size_inches(5.5, 2.5)
    sentiment_score_fig.tight_layout()  # Ensure tight layout
    sentiment_score_fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    sentiment_score_img = ImageReader(img_buffer)
    y_position = check_y_position(y_position, 20, 200)
    pdf.drawImage(sentiment_score_img, 100, y_position - 200, width=400, height=200)
    y_position -= 270

    # Add sentiment pie chart
    img_buffer = BytesIO()
    sentiment_pie_fig.set_size_inches(5, 3)
    sentiment_pie_fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    sentiment_pie_img = ImageReader(img_buffer)
    y_position = check_y_position(y_position, 50, 200)
    pdf.drawImage(sentiment_pie_img, 100, y_position - 200, width=400, height=200)
    y_position -= 250

    # Add word cloud
    # y_position = check_y_position(y_position, 50, 300)
    # wordcloud_img = ImageReader(wordcloud_buffer)
    # pdf.drawImage(wordcloud_img, 100, y_position - 300, width=400, height=300)
    # y_position -= 350

    # Add word cloud for positive sentiments
    y_position = check_y_position(y_position, 50, 300)  
    wordcloud_pos_img = ImageReader(wordcloud_pos_buffer)
    pdf.drawImage(wordcloud_pos_img, 100, y_position - 300, width=400, height=300)
    y_position -= 350

    # Add word cloud for negative sentiments
    y_position = check_y_position(y_position, 50, 300)
    wordcloud_neg_img = ImageReader(wordcloud_neg_buffer)
    pdf.drawImage(wordcloud_neg_img, 100, y_position - 300, width=400, height=300)
    y_position -= 350

    # Add top words bar chart for each topic
    num_topics = 5
    for topic_idx in range(min(num_topics, len(feature_names if is_lda else topic_model.get_topics()))):
        img_buffer = BytesIO()
        fig, ax = plt.subplots()
        
        if is_lda:
            topic_words = [feature_names[i] for i in topic_model.components_[topic_idx].argsort()[:-11:-1]]
            counts = topic_model.components_[topic_idx][topic_model.components_[topic_idx].argsort()[:-11:-1]]
        else:
            topic_words = [word for word, _ in topic_model.get_topic(topic_idx)]
            counts = [count for _, count in topic_model.get_topic(topic_idx)]
        
        ax.barh(topic_words, counts, color='skyblue')
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top Words in Topic {topic_idx + 1}')
        ax.invert_yaxis()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        topic_img = ImageReader(img_buffer)
        y_position = check_y_position(y_position, 50, 200)
        pdf.drawImage(topic_img, 100, y_position - 200, width=400, height=200)
        y_position -= 250
    
    pdf.save()
    buffer.seek(0)

    return buffer

def save_dataset_to_firestore(username, dataset_content, file_name, pdf_url, max_chunk_size=1048000):
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
    db = get_firestore_client()
    for idx, chunk in enumerate(csv_chunks):
        dataset_data = {
            "username": username,
            "timestamp": datetime.utcnow(),
            "chunk_id": idx,
            "dataset": chunk,
            "file_name": file_name,
            "pdf_url": pdf_url
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
            pdf_url = data.get("pdf_url", "No Report Available")

            if isinstance(timestamp, datetime):
                # Convert timestamp to the desired time zone (e.g., UTC+8)
                target_time_zone = pytz.timezone("Asia/Shanghai")
                local_timestamp = timestamp.astimezone(target_time_zone)
                formatted_timestamp = local_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_timestamp = timestamp

            # Append the file name, timestamp, and document ID to the history list
            history.append({
                "file_name": file_name,
                "upload_time": formatted_timestamp,
                "pdf_url": pdf_url,
                "doc_id": dataset.id
            })

        # Sort the history list by 'Upload Time' in descending order
        history = sorted(history, key=lambda x: x["upload_time"], reverse=True)

        # Display the history in a custom table
        st.write("Upload History:")

        # CSS for custom styling
        st.markdown("""
            <style>
                .history-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 1rem;
                }
                .history-table th, .history-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                .history-table th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                .history-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .history-table tr:hover {
                    background-color: #f1f1f1;
                }
                .history-table button {
                    padding: 5px 10px;
                    font-size: 14px;
                    cursor: pointer;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                .history-table .view-button {
                    background-color: #4CAF50;
                }
                .history-table .delete-button {
                    background-color: #f44336;
                }
            </style>
        """, unsafe_allow_html=True)

        # Create the table header
        st.markdown("""
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Dataset Name</th>
                        <th>Upload Time</th>
                        <th>View</th>
                        <th>Delete</th>
                    </tr>
                </thead>
                <tbody>
        """, unsafe_allow_html=True)

        for entry in history:
            col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
            with col1:
                st.write(entry["file_name"])
            with col2:
                st.write(entry["upload_time"])
            with col3:
                if entry["pdf_url"] != "No Report Available":
                    st.markdown(f'<a href="{entry["pdf_url"]}" target="_blank"><button class="view-button">View</button></a>', unsafe_allow_html=True)
                else:
                    st.write("No Report Available")
            with col4:
                if st.button("Delete", key=f'delete_{entry["doc_id"]}'):
                    st.session_state.confirm_delete = entry["doc_id"]
                    st.session_state.file_name_to_delete = entry["file_name"]
                    st.experimental_rerun()  # Rerun the app to refresh the state

        # End the table
        st.markdown("</tbody></table>", unsafe_allow_html=True)

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

def delete_history_entry(doc_id):
    try:
        db.collection("datasets").document(doc_id).delete()
        st.success("The file has been deleted successfully.")
    except Exception as e:
        st.error(f"An error occurred while deleting the file: {e}")

def perform_topic_modeling_lda(text_data):
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

    # Get topics
    topics = lda_model.components_

    return lda_model, feature_names, topics

def perform_topic_modeling_bertopic(text_data):
    # Initialize BERTopic model
    topic_model = BERTopic()

    # Fit the model on the text data
    topics, probs = topic_model.fit_transform(text_data)

    # Get the topic names and keywords
    topic_names = topic_model.get_topic_info()
    topic_keywords = {topic: keywords for topic, keywords in topic_model.get_topics().items()}

    return topic_model, topic_names, topic_keywords

def display_topic_words_lda(lda_model, feature_names, num_words=10):
    topic_word_data = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topic_word_data[f'Topic {topic_idx + 1}'] = topic_words
    
    cols = st.columns(len(topic_word_data))
    for idx, (topic, words) in enumerate(topic_word_data.items()):
        with cols[idx]:
            fig, ax = plt.subplots()
            counts = [lda_model.components_[list(topic_word_data.keys()).index(topic)][feature_names.tolist().index(word)] for word in words]
            ax.barh(words, counts, color='skyblue')
            ax.set_xlabel('Frequency')
            ax.set_title(f'Top Words in {topic}')
            ax.invert_yaxis()
            st.pyplot(fig)

def display_topic_words_bertopic(topic_model, topic_keywords, num_topics=5):
    topic_word_data = {}
    for topic, keywords in list(topic_keywords.items())[:num_topics]:
        words = [word for word, _ in keywords]
        topic_name = topic_model.get_topic_info().query(f"Topic == {topic}")["Name"].values[0]
        topic_word_data[topic_name] = words

    cols = st.columns(len(topic_word_data))
    for idx, (topic_name, words) in enumerate(topic_word_data.items()):
        with cols[idx]:
            fig, ax = plt.subplots()
            counts = [count for _, count in topic_keywords[topic]]
            ax.barh(words, counts, color='skyblue')
            ax.set_xlabel('Frequency')
            ax.set_title(f'Top Words in {topic_name}')  # Use topic name in title
            ax.invert_yaxis()
            st.pyplot(fig)

def generate_wordcloud(text):
    wordcloud = WordCloud().generate(text)
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# Function to cluster comments
def cluster_comments(dtm, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(dtm)
    clusters = kmeans.labels_
    return clusters

# Function to perform topic modeling using LDA
def lda_topic_modeling(dtm, n_topics=5):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    doc_topic_dists = lda.fit_transform(dtm)
    topic_term_dists = lda.components_ / lda.components_.sum(axis=1)[:, None]
    return lda, doc_topic_dists, topic_term_dists

# Visualize clusters
def visualize_clusters(analyzed_df):
    cluster_counts = analyzed_df['cluster'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
    plt.title('Comment Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Comments')
    st.pyplot(plt)

# Interactive topic modeling visualization
def interactive_topic_modeling(lda_model, dtm, vectorizer):
    vocab = {term: idx for idx, term in enumerate(vectorizer.get_feature_names_out())}
    term_frequency = dtm.sum(axis=0).A1  # Calculate term frequencies
    _, doc_topic_dists, topic_term_dists = lda_model
    panel = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, vocab, term_frequency=term_frequency, mds='tsne')
    pyLDAvis.save_html(panel, 'lda.html')
    with open('lda.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, height=800, scrolling=True)

# Heatmap of word frequencies by sentiment category
def plot_word_frequency_heatmap(df, num_words=20):
    positive_texts = ' '.join(df[df['Sentiment'] == 'Positive']['content'])
    negative_texts = ' '.join(df[df['Sentiment'] == 'Negative']['content'])
    neutral_texts = ' '.join(df[df['Sentiment'] == 'Neutral']['content'])

    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_words)
    pos_matrix = vectorizer.fit_transform([positive_texts])
    neg_matrix = vectorizer.fit_transform([negative_texts])
    neu_matrix = vectorizer.fit_transform([neutral_texts])

    pos_freq = pos_matrix.toarray().flatten()
    neg_freq = neg_matrix.toarray().flatten()
    neu_freq = neu_matrix.toarray().flatten()

    words = vectorizer.get_feature_names_out()

    freq_df = pd.DataFrame({'Positive': pos_freq, 'Negative': neg_freq, 'Neutral': neu_freq}, index=words)
    freq_df = freq_df.T

    plt.figure(figsize=(12, 6))
    sns.heatmap(freq_df, annot=True, cmap='coolwarm')
    plt.title('Word Frequencies by Sentiment Category')
    st.pyplot(plt)

# Sentiment scores by time (if timestamp is available)
def plot_sentiment_over_time(df, timestamp_column):
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df.set_index(timestamp_column, inplace=True)
    df['polarity'].resample('M').mean().plot(kind='line', figsize=(12, 6))
    plt.title('Sentiment Scores Over Time')
    plt.xlabel('Time')
    plt.ylabel('Average Polarity')
    st.pyplot(plt)

# main function
def comments_analyser():
    st.title("Comments Analyser")

    selected = st.sidebar.selectbox("Select Section", ["Upload Data", "View History"])
    

    if selected == "Upload Data":
        st.subheader("Upload Data")
        upl = st.file_uploader("Please upload dataset with ONE COLUMN containing text comments only.", type=["csv", "xlsx"])

        if upl:
            df = pd.read_csv(upl) if upl.name.endswith('.csv') else pd.read_excel(upl, engine='openpyxl')
            # st.write("First ten rows of Original Data:")
            # st.write(df.head(10))

            # Perform phrase extraction and sentiment analysis
            analyzed_df = analyze_dataframe(df)

            # st.write("First ten rows of Analyzed Data:")
            # st.write(analyzed_df.head(10))

            # st.write("Dataset is analyzed and saved successfully!")

            # Download analyzed data as CSV
            # @st.cache_data
            # def convert_df(df):
            #     return df.to_csv().encode('utf-8')

            # csv = convert_df(analyzed_df)

            # st.download_button(
            #     label="Download Analyzed Data as CSV",
            #     data=csv,
            #     file_name='analyzed_data.csv',
            #     mime='text/csv',
            # )

            # Perform data cleaning on the text column
            cleaned_column = analyzed_df[analyzed_df.columns[0]].apply(lambda x: cleantext.clean(x, clean_all=False, extra_spaces=True,
                                                                                stopwords=True, lowercase=True,
                                                                                numbers=True, punct=True))

            # Concatenate cleaned text
            # clean_text = ' '.join(cleaned_column)

            # Filter comments by sentiment
            positive_comments = analyzed_df[analyzed_df['Sentiment'] == 'Positive']
            negative_comments = analyzed_df[analyzed_df['Sentiment'] == 'Negative']

            positive_text = ' '.join(positive_comments[positive_comments.columns[0]])
            negative_text = ' '.join(negative_comments[negative_comments.columns[0]])

            st.markdown("<br><br>", unsafe_allow_html=True)

            st.header("Analysis Results")

            # Calculate the total number of comments
            total_comments = len(analyzed_df)
            st.write(f"Derived from {total_comments} comments")

            # Using st.metric for key metrics
            # col1, col2, col3 = st.columns(3)
            # col1.metric(label="Total Comments", value=total_comments)
            # col2.metric(label="Positive Comments", value=len(positive_comments))
            # col3.metric(label="Negative Comments", value=len(negative_comments))

            # Add CSS to style the columns
            st.markdown("""
                <style>
                .metric-container {
                    display: flex;
                    justify-content: space-between;
                    margin-top: 1rem;
                    margin-bottom: 1rem;
                }
                .metric-box {
                    background-color: #f9f9f9;
                    padding: 1rem;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    text-align: center;
                    width: 100%;
                    margin-right: 10px;
                    flex-grow: 1;
                }
                .metric-box h3 {
                    font-size: 1.5rem;
                    color: #333;
                    font-weight: 600;
                }
                </style>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-box">
                        <h3>{total_comments}</h3>
                        <p>Total Comments</p>   
                    </div>
                    <div class="metric-box">
                        <h3>{len(positive_comments)}</h3>
                        <p>Positive Comments</p>
                    </div>
                    <div class="metric-box">
                        <h3>{len(negative_comments)}</h3>
                        <p>Negative Comments</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)

            # col = st.columns((4,4), gap='small')
            col1, col2 = st.columns((1, 1))

            with col1: 
                st.write("First ten rows of Original Data:")
                st.write(df.head(10))

                st.markdown("<br><br>", unsafe_allow_html=True)
                # st.markdown("<br><br>", unsafe_allow_html=True)

                st.write("Word Cloud for Positive Sentiments")
                wordcloud_pos = WordCloud().generate(positive_text)
                fig_pos, ax_pos = plt.subplots()
                ax_pos.imshow(wordcloud_pos, interpolation='bilinear')
                ax_pos.axis("off")
                st.pyplot(fig_pos)
                wordcloud_pos_buffer = generate_wordcloud(positive_text)

                st.write("Word Cloud for Negative Sentiments")
                wordcloud_neg = WordCloud().generate(negative_text)
                fig_neg, ax_neg = plt.subplots()
                ax_neg.imshow(wordcloud_neg, interpolation='bilinear')
                ax_neg.axis("off")
                st.pyplot(fig_neg)
                wordcloud_neg_buffer = generate_wordcloud(negative_text)
                
                # Generate Word Cloud
                # wordcloud = WordCloud().generate(clean_text)
                # fig3, ax3 = plt.subplots()
                # ax3.imshow(wordcloud, interpolation='bilinear')
                # ax3.axis("off")
                # st.pyplot(fig3)
                # wordcloud_buffer = generate_wordcloud(clean_text) 
                
                
            with col2:
                st.write("First ten rows of Analyzed Data:")
                st.write(analyzed_df.head(10))
                @st.cache_data
                def convert_df(df):
                    return df.to_csv().encode('utf-8')

                csv = convert_df(analyzed_df)

                st.download_button(
                    label="Download Analyzed Data",
                    data=csv,
                    file_name='analyzed_data.csv',
                    mime='text/csv',
                )

                # st.markdown("<br><br>", unsafe_allow_html=True)

                # Percentage of Positive/Negative/Neutral Sentiment
                st.write("Percentage of Sentiment")
                sentiment_counts = analyzed_df['Sentiment'].value_counts(normalize=True) * 100
                labels = sentiment_counts.index
                sizes = sentiment_counts.values
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
                ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                # ax2.set_title('Percentage of Sentiment')
                st.pyplot(fig2)

                # Overall Sentiment Score
                st.write("Overall Sentiment Score")
                overall_score = analyzed_df['Polarity'].mean()
                overall_score_percentage = round((overall_score + 1) / 2 * 100, 2)  # Scale to [0, 100] for visualization

                fig1, ax1 = plt.subplots(figsize=(8, 2))
                ax1.barh(['Overall Sentiment Score'], [100], color='lightgray')  # Full percentage bar
                ax1.barh(['Overall Sentiment Score'], [overall_score_percentage], color='blue')  # Score percentage bar
                ax1.set_xlabel('Percentage')
                # ax1.set_title('Overall Sentiment Score')
                ax1.set_xlim(0, 100)
                ax1.invert_yaxis()  # Invert y-axis to have the bar extend from left to right
                ax1.text(overall_score_percentage + 2, 0, f'{overall_score_percentage}%', va='center')
                # Add sentiment indicators
                ax1.text(0, 0.4, '0%: Extremely Negative', ha='center', va='center', color='red', fontsize=8)
                ax1.text(50, 0.4, '50%: Neutral', ha='center', va='center', color='black', fontsize=8)
                ax1.text(100, 0.4, '100%: Extremely Positive', ha='center', va='center', color='green', fontsize=8)
                st.pyplot(fig1)

            # Word Frequency Heatmap by Sentiment Category
            # st.subheader("Heatmap of Word Frequencies by Sentiment Category")
            plot_word_frequency_heatmap(analyzed_df)
            
            # Topic Modeling Options
            topic_model_option = st.sidebar.selectbox("Select Topic Modeling Method", ["LDA", "BERTopic"])
            # selected = st.sidebar.selectbox("Select Section", ["Upload Data", "View History"])

            if topic_model_option == "LDA":
                lda_model, feature_names, topics = perform_topic_modeling_lda(cleaned_column)
                display_topic_words_lda(lda_model, feature_names)
                topic_model = lda_model
                is_lda = True
            elif topic_model_option == "BERTopic":
                topic_model, topic_names, topic_keywords = perform_topic_modeling_bertopic(cleaned_column)
                display_topic_words_bertopic(topic_model, topic_keywords)
                feature_names = []  # BERTopic doesn't use feature names in the same way
                is_lda = False
            
            # Generate the PDF report
            pdf_buffer = generate_pdf_report(analyzed_df, overall_score, sentiment_counts, wordcloud_pos_buffer, wordcloud_neg_buffer, fig1, fig2, topic_model, feature_names, is_lda)

            st.download_button(
                label="Download Report as PDF",
                data=pdf_buffer,
                file_name="sentiment_analysis_report.pdf",
                mime="application/pdf",
            )

            username = st.session_state.username
            dataset_content = upl.getvalue().decode('utf-8') if upl.name.endswith('.csv') else df.to_csv(index=False)
            file_name = upl.name
            # Upload PDF report to Firebase Storage and get the URL
            pdf_url = upload_pdf_to_storage(pdf_buffer, f"{username}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf")
            # Save dataset to Firestore with PDF URL
            save_dataset_to_firestore(username, dataset_content, file_name, pdf_url)

            # # Sentiment Scores Over Time (if timestamp is available)
            # if 'timestamp' in df.columns:
            #     st.subheader("Sentiment Scores Over Time")
            #     plot_sentiment_over_time(analyzed_df, 'timestamp')

            # # Initialize TfidfVectorizer
            # vectorizer = TfidfVectorizer(stop_words='english')
            # dtm = vectorizer.fit_transform(analyzed_df['content'])
            # feature_names = vectorizer.get_feature_names_out()

            # # Clustering comments
            # n_clusters = 3
            # analyzed_df['cluster'] = cluster_comments(dtm, n_clusters)

            # # Visualize clusters
            # st.header("Comment Clusters")
            # visualize_clusters(analyzed_df)

            # # Topic Modeling
            # n_topics = 5
            # lda_model, doc_topic_dists, topic_term_dists = lda_topic_modeling(dtm, n_topics)
            # interactive_topic_modeling((lda_model, doc_topic_dists, topic_term_dists), dtm, vectorizer)

    elif selected == "View History":
        st.subheader("View History")
        username = st.session_state.username
        datasets = get_datasets_for_user(username)
        if datasets:
            # display_datasets(datasets)
            view_upload_history(st.session_state.username)
        else:
            st.write("No datasets found.")

if __name__ == "__main__":
    comments_analyser()
