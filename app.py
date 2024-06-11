import streamlit as st
from auth import login, signup, logout, check_session_timeout
from comments_analyser import comments_analyser
from datetime import datetime, timedelta
# import datetime
import base64

st.set_page_config(page_title="Comments Analyser Web App", page_icon=":bar_chart:", layout="wide")

check_session_timeout()

# Initialize session state variables
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'useremail' not in st.session_state:
    st.session_state.useremail = ''
if "signedout" not in st.session_state:
    st.session_state["signedout"] = False
if 'signout' not in st.session_state:
    st.session_state['signout'] = False
if 'selected_section' not in st.session_state:
    st.session_state['selected_section'] = "Home"

# Function to set the background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def app():
    st.title('Comments Analyser System')

    # login/signup
    if not st.session_state["signedout"]:
        set_background("login.png")

        st.write("This system allows organizations to analyze comments or feedback data.")
        st.write("You can upload the data to see the visualization in dashboard view.")
        st.markdown("<br><br>", unsafe_allow_html=True)

        choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
        email = st.text_input('Email Address', key='email')
        password = st.text_input('Password', type='password', key='password')
        
        if choice == 'Sign up':
            signup_username = st.text_input("Enter your unique username", key='signup_username')
            st.button('Create my account', on_click=signup)
        else:
            st.button('Login', on_click=login)
    
    # when login is successful
    if st.session_state.signout:
        st.text('Name: ' + st.session_state.username)
        st.text('Email: ' + st.session_state.useremail)

        # Sidebar Navigation
        st.sidebar.title("Navigation")
        selected = st.sidebar.radio("Go to", ("Home", "Comments Analyser", "Contact", "Logout"))

        if selected == "Home":
            # First Section: Welcome Messages
            st.markdown(
                """
                <div style="text-align: center;">
                    <h2><b>Welcome to the Comments Analyzer System!</b></h2>
                    <p>This system allows you to analyze comments or feedback data.</p>
                    <p>You can upload your data to see the visualization in dashboard view.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Space between sections
            st.markdown("<br><br>", unsafe_allow_html=True)

            # Second Section: Steps to Use the System
            st.markdown(
                """
                <div style="text-align: left;">
                    <h3>Steps to Use the System:</h3>
                    <ol>
                        <li>Go to the "Upload Data" section to upload your comments or feedback data.</li>
                        <li>Once uploaded, the system will analyze the data and provide visualizations.</li>
                        <li>You can download the analyzed data as a CSV file.</li>
                        <li>Generate and download a comprehensive PDF report of the analysis.</li>
                        <li>View your previously uploaded history in the "View History" section.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True
            )

        if selected == "Contact":
            st.title("Contact")
            st.write("Email: commentsanalyser@cat405.my")

        if selected == "Logout":
            st.title("Logout")
            st.write("You have been logged out.")
            logout()

        if selected == "Comments Analyser":
            comments_analyser()

if __name__ == '__main__':
    app()
