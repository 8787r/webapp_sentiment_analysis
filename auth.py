import streamlit as st
import bcrypt
import firebase_admin
from firebase_admin import firestore, credentials, auth, storage
from datetime import datetime, timedelta
# import datetime
import time  # Import the time module

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccount.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'fyp-streamlitapp.appspot.com'  # Firebase Storage bucket
    })

# # Firestore client
db = firestore.client()

# Firebase Storage bucket
bucket = storage.bucket()

def get_firestore_client():
    return db

def get_storage_bucket():
    return bucket

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(hashed_password, user_password):
    return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password)

def login():
    email = st.session_state.email
    password = st.session_state.password
    try:
        user_ref = db.collection("user").where("email", "==", email).stream()
        user = next(user_ref, None)
        if user:
            user_data = user.to_dict()
            stored_password = user_data.get("password")
            if check_password(stored_password.encode('utf-8'), password):
                st.session_state.username = user_data.get("username")
                st.session_state.useremail = user_data.get("email")
                st.session_state.signedout = True
                st.session_state.signout = True
                # Set login time
                st.session_state.login_time = time.time()
            else:
                st.warning('Login Failed: Incorrect password')
        else:
            st.warning('Login Failed: User not found')
    except Exception as e:
        st.error(f'An error occurred: {e}')

def signup():
    email = st.session_state.email
    password = st.session_state.password
    username = st.session_state.signup_username
    try:
        # Check if the email already exists
        user_ref = db.collection("user").where("email", "==", email).stream()
        if next(user_ref, None):
            st.warning('Sign up failed: Email already exists')
            return

        hashed_password = hash_password(password)
        user_data = {
            "email": email,
            "password": hashed_password.decode('utf-8'),  # Store the hashed password
            "username": username
        }
        db.collection("user").document(username).set(user_data)
        st.success('Account created successfully!')
        st.markdown('Please Login using your email and password')
        st.balloons()
    except Exception as e:
        st.error(f'An error occurred: {e}')

def logout():
    st.session_state.signout = False
    st.session_state.signedout = False   
    st.session_state.username = ''
    st.experimental_rerun()

# Define session timeout duration (1 hour in this system)
SESSION_TIMEOUT_MINUTES = 60

def check_session_timeout():
    if 'last_activity' not in st.session_state:
        # If last activity timestamp not set, set it to the current time
        st.session_state.last_activity = datetime.now()
        return

    # Calculate time elapsed since last activity
    elapsed_time = datetime.now() - st.session_state.last_activity

    # If elapsed time exceeds session timeout duration, clear session
    if elapsed_time.total_seconds() > (SESSION_TIMEOUT_MINUTES * 60):
        # Clear session variables
        st.session_state.clear()
        # Redirect to login page or perform other actions as needed
        st.write("Session timed out. Please log in again.")
    else:
        # Update last activity timestamp to current time
        st.session_state.last_activity = datetime.now()
