# Web App Sentiment Analysis

## Overview
This project is a Python-based sentiment analysis web application built using Streamlit. It enables users to input textual comments and instantly receive sentiment classification results (Positive, Neutral, or Negative). The application demonstrates an end-to-end Natural Language Processing (NLP) analytics pipeline, from user input to sentiment inference and result visualization.

The project focuses on simplicity, explainability, and rapid deployment, making it suitable for internal analytics tools, demonstrations, and learning purposes.

## Table of Contents
- Overview
- Project Objectives
- System Architecture
- AI and NLP Approach
- Key Features
- Application Flow
- Usage
- Target Audience
- Limitations
- Technologies Used
- Future Enhancements

## Project Objectives
- To build an interactive web application for real-time sentiment analysis of text data.
- To demonstrate a complete NLP workflow from input preprocessing to sentiment classification.
- To provide an explainable and lightweight sentiment analysis solution.
- To showcase the integration of data analytics and AI techniques into a user-facing application.

## System Architecture
User (Browser)  
↓  
Streamlit Web Interface (app.py)  
↓  
Authentication Layer (auth.py)  
↓  
Sentiment Analysis Engine (comments_analyser.py)  
↓  
Rule-based NLP Sentiment Scoring  
↓  
Results Displayed in Web UI  

The application follows a modular architecture where the UI and application flow are managed by Streamlit, authentication logic is separated for access control, and sentiment analysis logic is encapsulated in a dedicated module.

## AI and NLP Approach
The sentiment analysis component uses a rule-based Natural Language Processing (NLP) approach rather than a trained machine learning model.

Processing steps include text preprocessing (normalization and basic cleaning), tokenization, sentiment scoring using a predefined sentiment lexicon, and classification into Positive, Neutral, or Negative categories.

This approach requires no training data, executes quickly, produces deterministic and explainable results, and is suitable for lightweight analytics and rapid prototyping. This project does not use deep learning or transformer-based models, as the design intentionally prioritizes transparency and interpretability.

## Key Features
- Web-based interactive interface using Streamlit
- Secure access through a simple authentication mechanism
- Real-time sentiment analysis
- Modular and maintainable Python codebase
- Easy local and cloud deployment

## Application Flow
1. User accesses the application through a web browser.
2. User logs in via the authentication interface.
3. Textual comments are entered into the application.
4. The sentiment analysis engine processes the input text.
5. Sentiment classification results are displayed instantly on the UI.

## Usage
1. Clone this repository:
   git clone https://github.com/8787r/webapp_sentiment_analysis.git
2. Navigate to the project directory:
   cd webapp_sentiment_analysis
3. Install the required dependencies:
   pip install -r requirements.txt
4. Run the Streamlit application:
   streamlit run app.py

The application will launch automatically in your default web browser.

## Target Audience
This project is suitable for data analysts, business analysts, students learning NLP and applied analytics, product and operations teams analyzing user feedback, developers exploring Streamlit-based web applications, and organizations seeking lightweight sentiment analysis solutions.

## Limitations
- Limited understanding of context and sarcasm
- Primarily designed for English-language text
- Rule-based approach does not adapt or learn from new data
- Not optimized for large-scale or high-concurrency environments

## Technologies Used
- Python: Core programming language
- Streamlit: Web application framework
- NLP Libraries: Lexicon-based sentiment analysis tools
- HTML/CSS (via Streamlit): User interface rendering

## Future Enhancements
- Integrate transformer-based NLP models (e.g., BERT, RoBERTa)
- Add batch sentiment analysis through CSV uploads
- Include sentiment distribution and trend visualizations
- Support multilingual sentiment analysis
- Introduce API-based backend for scalability and multi-user support
