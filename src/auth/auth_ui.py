"""
Authentication UI components for Streamlit.
"""

import streamlit as st
from src.auth.user_db import UserDatabase


def show_login_page(db: UserDatabase):
    """Display login/signup page."""

    st.markdown("<h2 style='text-align: center;'>Welcome to Modium</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888;'>Sign in to track your analysis history</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.markdown("### Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("LOGIN", use_container_width=True)

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    success, user_id = db.verify_user(username, password)
                    if success:
                        st.session_state['authenticated'] = True
                        st.session_state['user_id'] = user_id
                        st.session_state['username'] = username
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

    with tab2:
        st.markdown("### Create Account")
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email (optional)")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup = st.form_submit_button("CREATE ACCOUNT", use_container_width=True)

            if signup:
                if not new_username or not new_password:
                    st.error("Username and password are required")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, message = db.create_user(new_username, new_password, new_email)
                    if success:
                        st.success(message)
                        st.info("You can now login with your new account!")
                    else:
                        st.error(message)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666666; font-size: 0.9rem;'>Or continue without an account (analyses won't be saved)</p>", unsafe_allow_html=True)
    if st.button("CONTINUE AS GUEST", use_container_width=True):
        st.session_state['authenticated'] = False
        st.session_state['guest_mode'] = True
        st.rerun()


def show_user_menu(db: UserDatabase):
    """Display user menu in sidebar."""

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.get('username', 'Guest')}")

        if st.session_state.get('authenticated'):
            # Get user stats
            stats = db.get_user_stats(st.session_state['user_id'])

            st.markdown("---")
            st.markdown("#### Your Stats")
            st.metric("Total Analyses", stats['total_analyses'])

            if stats['total_analyses'] > 0:
                st.metric("Average Risk", f"{stats['average_risk_score']:.1f}")

            st.markdown("---")

            if st.button("📊 View History", use_container_width=True):
                st.session_state['page'] = 'history'
                st.rerun()

            if st.button("🏠 New Analysis", use_container_width=True):
                st.session_state['page'] = 'main'
                st.rerun()

            st.markdown("---")

            if st.button("Logout", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        else:
            st.info("You're in guest mode. Analyses won't be saved.")
            if st.button("Login / Sign Up", use_container_width=True):
                st.session_state.clear()
                st.rerun()


def show_history_page(db: UserDatabase):
    """Display user's analysis history."""

    st.markdown("<h2 style='text-align: center;'>Analysis History</h2>", unsafe_allow_html=True)
    st.markdown("---")

    user_id = st.session_state.get('user_id')
    if not user_id:
        st.warning("Please login to view history")
        return

    analyses = db.get_user_analyses(user_id)

    if not analyses:
        st.info("No analyses yet. Upload an image to get started!")
        return

    st.markdown(f"### Total Analyses: {len(analyses)}")
    st.markdown("---")

    for i, analysis in enumerate(analyses):
        with st.expander(f"📅 {analysis['timestamp']} - Risk: {analysis['risk_level']}", expanded=(i == 0)):
            col1, col2 = st.columns([1, 2])

            with col1:
                # Display image
                if analysis['image_data']:
                    import base64
                    image_bytes = base64.b64decode(analysis['image_data'])
                    st.image(image_bytes, caption="Analyzed Image")

            with col2:
                # Display results
                st.markdown(f"**Prediction:** {analysis['predicted_class'].capitalize()}")
                st.markdown(f"**Confidence:** {analysis['confidence']:.1%}")
                st.markdown(f"**Risk Score:** {analysis['risk_score']:.1f}/100")
                st.markdown(f"**Risk Level:** {analysis['risk_level']}")

                st.markdown("**Visual Features:**")
                features = analysis['visual_features']
                for key, value in features.items():
                    if value is not None and key not in ['evolution_note', 'evolution_recommendation']:
                        st.progress(value, text=f"{key.replace('_', ' ').title()}: {value:.2f}")

                st.markdown("**Recommendations:**")
                for rec in analysis['recommendations']:
                    st.markdown(f"- {rec}")
