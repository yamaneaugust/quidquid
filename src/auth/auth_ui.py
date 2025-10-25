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
    if st.button("← BACK TO APP", use_container_width=True):
        st.session_state['authenticated'] = False
        st.session_state['guest_mode'] = True
        st.session_state['page'] = 'main'
        st.rerun()


def show_signup_popup():
    """Show a popup encouraging guests to sign up."""
    st.info("""
    **Create a free account to save your analysis history**

    With an account you can:
    - Save your analysis history
    - Track changes over time
    - Access past reports anytime

    It only takes 30 seconds!
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Account", use_container_width=True):
            st.session_state['page'] = 'login'
            st.rerun()
    with col2:
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state['dismiss_signup'] = True
            st.rerun()


def show_user_menu(db: UserDatabase):
    """Display user menu in sidebar."""

    with st.sidebar:
        st.markdown(f"### {st.session_state.get('username', 'Guest')}")

        if st.session_state.get('authenticated'):
            # Get user stats
            stats = db.get_user_stats(st.session_state['user_id'])

            st.markdown("---")
            st.markdown("#### Your Stats")
            st.metric("Total Analyses", stats['total_analyses'])

            if stats['total_analyses'] > 0:
                st.metric("Average Risk", f"{stats['average_risk_score']:.1f}")

            st.markdown("---")

            if st.button("View History", use_container_width=True):
                st.session_state['page'] = 'history'
                st.rerun()

            if st.button("Compare Analyses", use_container_width=True):
                st.session_state['page'] = 'compare'
                st.rerun()

            if st.button("New Analysis", use_container_width=True):
                st.session_state['page'] = 'main'
                st.rerun()

            st.markdown("---")

            if st.button("Logout", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        else:
            st.info("Create an account to save your analysis history")
            if st.button("Sign Up / Login", use_container_width=True):
                st.session_state['page'] = 'login'
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
        with st.expander(f"{analysis['timestamp']} - Risk: {analysis['risk_level']}", expanded=(i == 0)):
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


def show_comparison_page(db: UserDatabase):
    """Display comparison page for tracking changes over time."""

    st.markdown("<h2 style='text-align: center;'>Compare Analyses</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888;'>Track changes in your lesions over time</p>", unsafe_allow_html=True)
    st.markdown("---")

    user_id = st.session_state.get('user_id')
    if not user_id:
        st.warning("Please login to compare analyses")
        return

    analyses = db.get_user_analyses(user_id)

    if len(analyses) < 2:
        st.info("You need at least 2 analyses to compare. Upload more images to track changes over time!")
        return

    st.markdown("### Select Analyses to Compare")
    st.markdown("<p style='color: #888888; font-size: 0.9rem;'>Choose 2-4 analyses to compare side-by-side</p>", unsafe_allow_html=True)

    # Create selection options
    analysis_options = {}
    for analysis in analyses:
        label = f"{analysis['timestamp']} - {analysis['predicted_class'].capitalize()} ({analysis['risk_level']})"
        analysis_options[label] = analysis

    # Multi-select
    selected_labels = st.multiselect(
        "Select analyses",
        options=list(analysis_options.keys()),
        max_selections=4,
        label_visibility="collapsed"
    )

    if len(selected_labels) < 2:
        st.info("Please select at least 2 analyses to compare")
        return

    if st.button("COMPARE SELECTED", use_container_width=True):
        selected_analyses = [analysis_options[label] for label in selected_labels]

        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Comparison Results</h3>", unsafe_allow_html=True)
        st.markdown("---")

        # Side-by-side comparison
        cols = st.columns(len(selected_analyses))

        for i, (col, analysis) in enumerate(zip(cols, selected_analyses)):
            with col:
                st.markdown(f"#### Analysis {i+1}")
                st.markdown(f"**Date:** {analysis['timestamp']}")

                # Display image
                if analysis['image_data']:
                    import base64
                    image_bytes = base64.b64decode(analysis['image_data'])
                    st.image(image_bytes, use_column_width=True)

                # Risk score
                risk_colors = {
                    "Low": "#28a745",
                    "Moderate": "#ffc107",
                    "High": "#fd7e14",
                    "Very High": "#dc3545"
                }
                risk_color = risk_colors.get(analysis['risk_level'], "#ffffff")

                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background-color: #1a1a1a; border-radius: 4px; margin: 1rem 0;'>
                    <h2 style='color: {risk_color}; font-size: 2rem; margin: 0;'>{analysis['risk_score']:.0f}</h2>
                    <p style='color: #888888; font-size: 0.8rem; margin: 0;'>RISK SCORE</p>
                    <p style='color: {risk_color}; margin-top: 0.5rem; font-size: 0.9rem;'>{analysis['risk_level']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Prediction
                st.markdown(f"**Prediction:** {analysis['predicted_class'].capitalize()}")
                st.markdown(f"**Confidence:** {analysis['confidence']:.1%}")

        st.markdown("---")

        # Trend analysis
        st.markdown("### Trend Analysis")

        # Risk score trend
        risk_scores = [a['risk_score'] for a in selected_analyses]
        risk_change = risk_scores[-1] - risk_scores[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("First Risk Score", f"{risk_scores[0]:.1f}")
        with col2:
            st.metric("Latest Risk Score", f"{risk_scores[-1]:.1f}", delta=f"{risk_change:+.1f}")
        with col3:
            if risk_change > 10:
                st.error("**Increasing Risk** - Consult a dermatologist")
            elif risk_change > 0:
                st.warning("**Slight Increase** - Monitor closely")
            elif risk_change < -10:
                st.success("**Decreasing Risk** - Positive trend")
            else:
                st.info("**Stable** - Continue monitoring")

        st.markdown("---")

        # ABCDE criteria comparison
        st.markdown("### ABCDE Criteria Comparison")

        abcde_keys = ['asymmetry', 'border_irregularity', 'color_variation', 'diameter_score']
        abcde_labels = {
            'asymmetry': 'Asymmetry',
            'border_irregularity': 'Border Irregularity',
            'color_variation': 'Color Variation',
            'diameter_score': 'Diameter'
        }

        for key in abcde_keys:
            st.markdown(f"**{abcde_labels[key]}:**")
            cols = st.columns(len(selected_analyses))
            for i, (col, analysis) in enumerate(zip(cols, selected_analyses)):
                with col:
                    value = analysis['visual_features'].get(key, 0)
                    if value is not None:
                        st.progress(value, text=f"Analysis {i+1}: {value:.2f}")

        st.markdown("---")

        # Recommendations
        st.markdown("### Monitoring Recommendations")

        if risk_change > 10:
            st.error("""
            **Urgent Attention Required**

            The risk score has increased significantly. This could indicate:
            - Growth or changes in the lesion
            - Evolution of concerning features

            **Recommended Action:** Schedule a dermatologist appointment immediately.
            """)
        elif risk_change > 5:
            st.warning("""
            **Increased Monitoring Recommended**

            The risk score has increased moderately.

            **Recommended Action:**
            - Take another photo in 2-4 weeks
            - Schedule dermatologist appointment if changes continue
            - Document any symptoms (itching, bleeding, etc.)
            """)
        elif risk_change < -5:
            st.success("""
            **Positive Trend**

            The risk score has decreased.

            **Recommended Action:**
            - Continue regular monitoring
            - Maintain current skincare routine
            - Take another photo in 3-6 months
            """)
        else:
            st.info("""
            **Stable Lesion**

            The risk score has remained relatively stable.

            **Recommended Action:**
            - Continue routine monitoring
            - Take another photo in 3 months
            - Watch for any sudden changes
            """)

        st.markdown("---")
        st.error("""
        **Medical Disclaimer**

        This comparison tool is for educational and documentation purposes only.
        It is NOT a substitute for professional medical diagnosis.
        Always consult a qualified dermatologist for medical decisions.
        """)
