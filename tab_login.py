import streamlit as st
from config.Authenticator import Authenticator
from mmlogger import setup_logger
from src.MMTelemetryPayload.payload import Payload
import yaml


# Set up the logger
logger = setup_logger('tab_login')


def initialize_session(authenticator):
    """Initialize session variables if not already set."""
    # Initialize session state variables
    logger.info(st.session_state)

    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'show_feedback_form' not in st.session_state:
        st.session_state.show_feedback_form = False

    if "code" in st.query_params and 'authenticated' not in st.session_state:
        try:
            authenticator.check_auth_code(st.query_params)
            user_info = authenticator.get_user_info().user_info
            if user_info.get('email'):
                set_session(authenticator)
            logger.info("Authentication code found and checked.")
        except Exception as e:
            logger.error(f"Error checking authentication code: {e}")
        st.query_params.clear()
        st.rerun()

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.display_name = ""
        st.session_state.profile_pic_url = ""
        st.session_state.email_address = ""
        st.session_state.credentials = None
    logger.info(f"Session initialized. {[key for key in st.session_state]}")


def clear_session():
    """Clear session state variables."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    logger.info("Session cleared.")


def set_session(authenticator):
    """Set session state with user info."""
    try:
        user_info = authenticator.get_user_info().user_info
        st.session_state.display_name = user_info.get('name')
        st.session_state.email_address = user_info.get('email')
        st.session_state.profile_pic_url = user_info.get('picture')
        st.session_state.authenticated = True
        logger.info(f"User session set for {st.session_state.display_name}.")
    except Exception as e:
        logger.error(f"Error setting user session: {e}")


def authenticate_user(authenticator):
    """Authenticate user with Google OAuth."""
    with st.empty():
        if not st.session_state.authenticated:
            if authenticator.credentials and authenticator.credentials.valid:
                set_session(authenticator)
                display_user_info()
            elif st.button('Login with Google', type="primary"):
                payload = Payload(app_name="MMM-AUTOMATION", google_id="", username="", email_id="",
                                  logger=logger)
                payload.add_email_id(st.session_state.email_address)
                payload.add_username(st.session_state.display_name)
                payload.add_action(action_name="Login", action_attributes={"status": 'started'})
                try:
                    authenticator.get_credentials()
                    if authenticator.auth_url:
                        # Display login button and redirect to Google OAuth
                        authenticator.initiate_flow()
                        if authenticator.auth_url:
                            st.markdown(f"""
                                <meta http-equiv="refresh" content = "1;url={authenticator.auth_url}" />
                                If it\'s not redirected within a second: <a href="{authenticator.auth_url}" target="_top" onload="window.location.href=this.href;">Click here</a>
                            """, unsafe_allow_html=True)
                        logger.info("Redirecting user to Google OAuth.")
                        st.stop()
                except Exception as e:
                    logger.error(f"Error initiating Google OAuth flow: {e}")
                payload.send_payload()
                set_session(authenticator)
                st.rerun()


def display_user_info():
    """Display the logged-in user's profile."""
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; flex-direction: column;">
        <div style="display: flex;">
            <img src="{st.session_state.profile_pic_url}" width="90" style="border-radius: 50%; margin-right: 10px;">
            <h1 style="margin: 0;">Welcome <a href="mailto:{st.session_state.email_address}" style="text-decoration: none;">{st.session_state.display_name}</a></h1>
        </div>
        <p style="color: grey;">You are logged in with email {st.session_state.email_address}</p>
    </div>
    <style>[data-testid="stHeaderActionElements"]{{ display: none; }}</style>
    """, unsafe_allow_html=True)
    logger.info(f"Displayed user info for {st.session_state.display_name}.")


def handle_login(authenticator):
    """Main function to handle login and user session."""
    initialize_session(authenticator)
    if st.session_state.authenticated:
        display_user_info()
    else:
        authenticate_user(authenticator)

def fileconfig(path="config.yaml"):

    with open(path,"r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    authenticator = Authenticator()
    handle_login(authenticator)
    pass
