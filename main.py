# import libraries
import pickle
from pathlib import Path
from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
from nltk.corpus import stopwords
nltk.download('stopwords')

st.set_page_config(page_title="Sentiment Analysis Web App", page_icon=":bar_chart:",layout="wide")

# user authentication
names = ["Peter Parker", "Rebecca Miller"]
usernames = ["pparker", "rmiller"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "sales_dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    # navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Text Analysis", "Contact", "Logout"],
        icons=["house", "search", "envelope", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Home":
        st.title(f"{selected}")
    if selected == "Text Analysis":
        st.title(f"{selected}")
    if selected == "Contact":
        st.title(f"{selected}")
    if selected == "Logout":
        authenticator.logout("Logout", "main")

    if selected == "Text Analysis":

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

        # Main function to run the Streamlit app
        def main():

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
                    st.write("Overall Sentiment Score:", round(overall_score, 2))

                    # Percentage of Positive/Negative/Neutral Sentiment
                    sentiment_counts = analyzed_df['Sentiment'].value_counts(normalize=True) * 100
                    st.write("Percentage of Positive Sentiment:", round(sentiment_counts.get('Positive', 0), 2), "%")
                    st.write("Percentage of Negative Sentiment:", round(sentiment_counts.get('Negative', 0), 2), "%")
                    st.write("Percentage of Neutral Sentiment:", round(sentiment_counts.get('Neutral', 0), 2), "%")

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

                    # Top Ten Words with Highest Frequency
                    word_frequencies = pd.Series(clean_text.split()).value_counts()[:10]
                    st.write("Top Ten Words with Highest Frequency:")
                    st.write(word_frequencies) 

                    # # Themes discovery
                    # # Split cleaned text into words
                    # words = clean_text.split()
                    # # Count the frequency of each word
                    # word_freq = Counter(words)
                    # # Get the top themes with their frequencies
                    # top_themes = word_freq.most_common(10)
                    # # Display the top themes with their frequencies
                    # st.write("Top Themes with Frequency:")
                    # for theme, frequency in top_themes:
                    #     st.write(f"{theme}: {frequency}")                 

                    ########################### generate report ###########################

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
                        
                        # Overall Sentiment Score
                        pdf.drawString(100, 750, f"Overall Sentiment Score: {overall_score}")
                        
                        # Percentage of Sentiments
                        pdf.drawString(100, 730, f"Percentage of Positive Sentiment: {sentiment_counts.get('Positive', 0)} %")
                        pdf.drawString(100, 710, f"Percentage of Negative Sentiment: {sentiment_counts.get('Negative', 0)} %")
                        pdf.drawString(100, 690, f"Percentage of Neutral Sentiment: {sentiment_counts.get('Neutral', 0)} %")
                        
                        # Word Cloud
                        wordcloud_img = ImageReader(wordcloud_buffer)  # Convert BytesIO to ImageReader
                        pdf.drawImage(wordcloud_img, 100, 500, width=400, height=300)

                        # Top Ten Words with Highest Frequency
                        pdf.drawString(100, 450, "Top Ten Words with Highest Frequency:")
                        y_position = 430
                        for word, frequency in word_frequencies.items():
                            pdf.drawString(100, y_position, f"{word}: {frequency}")
                            y_position -= 20
                        
                        pdf.save()
                        
                        buffer.seek(0)
                        return buffer

                    # Generate the WordCloud
                    wordcloud_buffer = generate_wordcloud(clean_text)

                    # Function to generate and download the report
                    def generate_and_download_report():
                        # Show spinner while generating report
                        with st.spinner("Generating report..."):
                            # Call the function to generate the report
                            pdf_buffer = generate_pdf_report(overall_score, sentiment_counts, wordcloud_buffer, word_frequencies)
                            
                        # Hide spinner and offer the PDF report as a download link
                        st.success("Report generated successfully!")
                        st.download_button(
                            label="Download Report as PDF",
                            data=pdf_buffer,
                            file_name="sentiment_analysis_report.pdf",
                            mime="application/pdf",
                        )
                        
                        # Stop the Streamlit app's execution
                        st.experimental_stop()

                    # Offer the option to download the report
                    if st.button("Download Report"):
                        generate_and_download_report()

            elif selected == "View History":
                st.header("View History")
                st.write("Coming soon...")

        if __name__ == "__main__":
            main()