import streamlit as st
import time
import os

def main():
    st.set_page_config(
        page_title="HealthBot - Login",
        page_icon="🔐",
        layout="centered"
    )
    
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .login-header {
            text-align: center;
            margin-bottom: 40px;
        }
        .login-card {
            background: white;
            padding: 40px 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        .stButton>button {
            border-radius: 25px;
            padding: 12px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            margin-top: 10px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 12px 15px;
            margin-bottom: 15px;
        }
        .register-link {
            text-align: center;
            margin-top: 20px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <h1 style='font-size: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;'>🏥 HealthBot</h1>
                <p style='color: #666; font-size: 1.1rem;'>Your AI Health Assistant</p>
            </div>
            
            <div class="login-card">
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'users' not in st.session_state:
        st.session_state.users = {}
    if 'gmail_users' not in st.session_state:
        st.session_state.gmail_users = {}
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # If user is already logged in, redirect to app
    if st.session_state.get('logged_in', False):
        st.success("✅ Already logged in! Redirecting...")
        time.sleep(1)
        # Use JavaScript redirect instead of switch_page
        st.markdown(
            """
            <script>
                window.location.href = "http://localhost:8501/app";
            </script>
            """,
            unsafe_allow_html=True
        )
        return

    # Show registration form if toggle is True
    if st.session_state.show_register:
        show_registration_form()
    else:
        show_login_form()
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Demo credentials info
    st.markdown("""
        <div style='text-align: center; margin-top: 20px; color: #666;'>
            <p><strong>Demo Credentials:</strong></p>
            <p>Username: <code>user</code> | Password: <code>password123</code></p>
            <p>Username: <code>admin</code> | Password: <code>admin123</code></p>
            <p>Username: <code>test</code> | Password: <code>test123</code></p>
            <p>Any Gmail | Password: <code>gmail123</code></p>
        </div>
    """, unsafe_allow_html=True)

def show_registration_form():
    """Show registration form"""
    st.subheader("📝 Create Account")
    
    with st.form("register_form"):
        register_method = st.radio(
            "Choose account type:",
            ["Username/Password", "Gmail"],
            horizontal=True,
            key="register_method"
        )
        
        if register_method == "Username/Password":
            new_username = st.text_input("👤 Choose Username", placeholder="Enter your username", key="reg_username")
            new_password = st.text_input("🔒 Create Password", type="password", placeholder="Create a password", key="reg_password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password", key="reg_confirm")
            
        else:  # Gmail registration
            new_gmail = st.text_input("📧 Gmail Address", placeholder="Enter your Gmail address", key="reg_gmail")
            new_password = st.text_input("🔒 Create Password", type="password", placeholder="Create a password", key="reg_gmail_password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password", key="reg_gmail_confirm")
        
        col1, col2 = st.columns(2)
        with col1:
            register_submitted = st.form_submit_button("📝 Register", use_container_width=True)
        with col2:
            back_to_login = st.form_submit_button("🔙 Back to Login", use_container_width=True)
        
        if back_to_login:
            st.session_state.show_register = False
            st.rerun()
        
        if register_submitted:
            if register_method == "Username/Password":
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        # Check if username already exists
                        if new_username in st.session_state.users:
                            st.error("❌ Username already exists! Please choose a different username.")
                        else:
                            # Store user data
                            st.session_state.users[new_username] = new_password
                            st.success("✅ Account created successfully! Please login with your new credentials.")
                            time.sleep(2)
                            st.session_state.show_register = False
                            st.rerun()
                    else:
                        st.error("❌ Passwords do not match! Please try again.")
                else:
                    st.error("❌ Please fill in all fields!")
            
            else:  # Gmail registration
                if new_gmail and new_password and confirm_password:
                    if "@gmail.com" not in new_gmail:
                        st.error("❌ Please enter a valid Gmail address!")
                    elif new_password == confirm_password:
                        # Check if Gmail already exists
                        if new_gmail in st.session_state.gmail_users:
                            st.error("❌ Gmail already registered! Please use a different Gmail.")
                        else:
                            # Store Gmail user data
                            st.session_state.gmail_users[new_gmail] = new_password
                            st.success("✅ Account created successfully! Please login with your Gmail.")
                            time.sleep(2)
                            st.session_state.show_register = False
                            st.rerun()
                    else:
                        st.error("❌ Passwords do not match! Please try again.")
                else:
                    st.error("❌ Please fill in all fields!")

def show_login_form():
    """Show login form"""
    st.subheader("🔐 Login to HealthBot")
    
    with st.form("login_form"):
        login_method = st.radio(
            "Choose login method:",
            ["Username/Password", "Gmail"],
            horizontal=True,
            key="login_method"
        )
        
        if login_method == "Username/Password":
            username = st.text_input("👤 Username", placeholder="Enter your username", key="login_username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_password")
            
        else:  # Gmail login
            gmail = st.text_input("📧 Gmail", placeholder="Enter your Gmail address", key="login_gmail")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_gmail_password")
        
        login_submitted = st.form_submit_button("🚀 Login", use_container_width=True)
        
        if login_submitted:
            if login_method == "Username/Password":
                if username and password:
                    # Check demo credentials first
                    demo_credentials = {
                        "user": "password123",
                        "admin": "admin123",
                        "test": "test123"
                    }
                    
                    # Check registered users
                    registered_users = st.session_state.users
                    
                    if username in demo_credentials and password == demo_credentials[username]:
                        handle_successful_login(username, "username")
                    elif username in registered_users and registered_users[username] == password:
                        handle_successful_login(username, "username")
                    else:
                        st.error("❌ Invalid username or password!")
                else:
                    st.error("❌ Please fill in all fields!")
            
            else:  # Gmail login
                if gmail and password:
                    # Check demo Gmail
                    demo_gmail_password = "gmail123"
                    
                    # Check registered Gmail users
                    registered_gmail_users = st.session_state.gmail_users
                    
                    if "@gmail.com" in gmail and password == demo_gmail_password:
                        handle_successful_login(gmail, "gmail")
                    elif gmail in registered_gmail_users and registered_gmail_users[gmail] == password:
                        handle_successful_login(gmail, "gmail")
                    else:
                        st.error("❌ Invalid Gmail or password!")
                else:
                    st.error("❌ Please fill in all fields!")
    
    # Register button below login form
    st.markdown("---")
    st.markdown('<div class="register-link">', unsafe_allow_html=True)
    st.write("Don't have an account?")
    if st.button("📝 Create New Account", use_container_width=True, key="register_btn"):
        st.session_state.show_register = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def handle_successful_login(username, user_type):
    """Handle successful login"""
    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.user_type = user_type
    st.success(f"✅ Welcome {'back' if user_type == 'username' else ''}, {username}!")
    
    # Show redirect message
    st.info("🔄 Redirecting to HealthBot...")
    
    # Use JavaScript to redirect
    st.markdown(
        """
        <script>
            setTimeout(function() {
                window.location.href = window.location.origin + "/app";
            }, 1500);
        </script>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
