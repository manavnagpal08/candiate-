import streamlit as st
import json
import os
import pandas as pd
import urllib.parse
from datetime import datetime, date # Import date specifically
import bcrypt # For secure password hashing

# Import all necessary functions and constants from the new screener_logic.py file
from screener_logic import (
    global_sentence_model, global_ml_model,
    process_single_resume_for_candidate_app,
    generate_certificate_html, generate_certificate_pdf, send_certificate_email,
    APP_BASE_URL, CERTIFICATE_HOSTING_URL, HR_APP_URL
)

# --- Streamlit Page Configuration (MUST be at the very top) ---
st.set_page_config(page_title="Candidate Screener – AI Resume Match", layout="centered", page_icon="📄")

# --- Local File Paths ---
USER_DB_FILE_CANDIDATE = "candidate_users.json" # Separate user database for candidates
CANDIDATE_HISTORY_FILE = "candidate_screening_history.json"
CANDIDATE_LEADERBOARD_FILE = "candidate_leaderboard.json"
JD_FOLDER = "data" # Folder where JDs are stored
STYLE_CSS_FILE = "style.css" # Path to the CSS file

# Ensure the JD folder exists and create a dummy JD if empty
if not os.path.exists(JD_FOLDER):
    os.makedirs(JD_FOLDER)
    # Create a dummy JD file if the folder was just created and is empty
    dummy_jd_path = os.path.join(JD_FOLDER, "sample_software_engineer_jd.txt")
    if not os.path.exists(dummy_jd_path):
        with open(dummy_jd_path, "w") as f:
            f.write("""
            Job Title: Software Engineer

            Company: InnovateTech Solutions

            Location: Remote

            About Us:
            InnovateTech Solutions is a fast-growing tech company specializing in cloud-native applications and AI-driven platforms. We are looking for passionate and skilled Software Engineers to join our dynamic team and contribute to cutting-edge projects.

            Responsibilities:
            - Design, develop, test, deploy, and maintain scalable and robust software solutions.
            - Write clean, maintainable, and efficient code in Python, Java, or Go.
            - Collaborate with cross-functional teams to define, design, and ship new features.
            - Participate in code reviews to ensure code quality and share knowledge.
            - Troubleshoot, debug, and upgrade existing systems.
            - Implement and manage CI/CD pipelines.
            - Work with cloud platforms such as AWS, Azure, or Google Cloud.
            - Utilize containerization technologies like Docker and Kubernetes.

            Requirements:
            - Bachelor's or Master's degree in Computer Science, Engineering, or a related field.
            - 3+ years of professional software development experience.
            - Strong proficiency in Python or Java.
            - Experience with RESTful APIs and microservices architecture.
            - Familiarity with relational and NoSQL databases (e.g., PostgreSQL, MongoDB).
            - Solid understanding of data structures, algorithms, and object-oriented design.
            - Experience with version control systems (Git).
            - Excellent problem-solving and communication skills.

            Preferred Qualifications:
            - Experience with front-end frameworks (React, Angular, Vue.js).
            - Knowledge of machine learning concepts and libraries (TensorFlow, PyTorch).
            - Contributions to open-source projects.
            - Experience with agile development methodologies.
            """)
        st.info(f"Created a sample JD: {dummy_jd_path} in the '{JD_FOLDER}' folder.")


# --- User Authentication and Data Management ---

def load_candidate_users_local():
    """Loads candidate user data from a local JSON file."""
    if os.path.exists(USER_DB_FILE_CANDIDATE):
        with open(USER_DB_FILE_CANDIDATE, "r") as f:
            return json.load(f)
    return {}

def save_candidate_users_local(users):
    """Saves candidate user data to a local JSON file."""
    with open(USER_DB_FILE_CANDIDATE, "w") as f:
        json.dump(users, f, indent=4)

def load_candidate_screening_history_local():
    """Loads candidate screening history from a local JSON file."""
    if os.path.exists(CANDIDATE_HISTORY_FILE):
        with open(CANDIDATE_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_candidate_screening_result_local(user_email, result):
    """Saves a single screening result to the candidate's local history."""
    history = load_candidate_screening_history_local()
    result_with_timestamp = result.copy()
    result_with_timestamp['timestamp'] = datetime.now().isoformat()
    result_with_timestamp['user_email'] = user_email # Link result to user
    history.append(result_with_timestamp)
    with open(CANDIDATE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_top_candidates_leaderboard_local():
    """Loads top candidates data from a local JSON file."""
    if os.path.exists(CANDIDATE_LEADERBOARD_FILE):
        with open(CANDIDATE_LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    return []

def update_top_candidates_leaderboard_local(candidate_name, score, user_email):
    """
    Updates the local top candidates leaderboard.
    Keeps only the top 10 unique candidates by name.
    """
    leaderboard = load_top_candidates_leaderboard_local()
    
    # Remove existing entry for this candidate to update their score
    leaderboard = [entry for entry in leaderboard if entry.get('Candidate Name') != candidate_name]

    # Add the new entry
    new_entry = {
        "Candidate Name": candidate_name,
        "Score (%)": score,
        "Timestamp": datetime.now().isoformat(),
        "user_email": user_email # Store user email for accountability
    }
    leaderboard.append(new_entry)

    # Sort by score (descending) and timestamp (descending for ties)
    leaderboard.sort(key=lambda x: (x.get('Score (%)', 0), x.get('Timestamp', '')), reverse=True)
    
    # Keep only the top 10 unique candidates
    unique_candidates = []
    seen_names = set()
    for entry in leaderboard:
        if entry.get('Candidate Name') not in seen_names:
            unique_candidates.append(entry)
            seen_names.add(entry.get('Candidate Name'))
        if len(unique_candidates) >= 10: # Keep top 10 unique
            break
    
    with open(CANDIDATE_LEADERBOARD_FILE, "w") as f:
        json.dump(unique_candidates, f, indent=4)


def candidate_login_section():
    """Handles user login and registration for candidates."""
    st.sidebar.subheader("🔒 Candidate Login / Register")

    if 'candidate_authenticated' not in st.session_state:
        st.session_state.candidate_authenticated = False
    if 'candidate_username' not in st.session_state:
        st.session_state.candidate_username = None

    if st.session_state.candidate_authenticated:
        return True

    login_tab, register_tab = st.sidebar.tabs(["Login", "Register"])

    with login_tab:
        st.markdown("#### Existing User Login") # Use markdown for sub-headers within tabs
        username = st.text_input("Email", key="login_email_candidate").strip()
        password = st.text_input("Password", type="password", key="login_password_candidate").strip()

        if st.button("Login", key="login_button_candidate"):
            users = load_candidate_users_local()
            if username in users:
                hashed_password = users[username]['password']
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                    st.session_state.candidate_authenticated = True
                    st.session_state.candidate_username = username
                    st.sidebar.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.sidebar.error("Incorrect password.")
            else:
                st.sidebar.error("User not found. Please register or check your email.")

    with register_tab:
        st.markdown("#### New User Registration") # Use markdown for sub-headers within tabs
        new_username = st.text_input("Email (for registration)", key="register_email_candidate").strip()
        new_password = st.text_input("Password (for registration)", type="password", key="register_password_candidate").strip()
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_candidate").strip()

        if st.button("Register", key="register_button_candidate"):
            if new_username and new_password and confirm_password:
                if new_password == confirm_password:
                    users = load_candidate_users_local()
                    if new_username not in users:
                        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                        users[new_username] = {'password': hashed_password}
                        save_candidate_users_local(users)
                        st.sidebar.success("Registration successful! You can now log in.")
                    else:
                        st.sidebar.warning("Email already registered. Please log in.")
                else:
                    st.sidebar.error("Passwords do not match.")
            else:
                st.sidebar.error("Please fill in all registration fields.")
    return st.session_state.candidate_authenticated

# --- Helper function to load JDs ---
def load_jds_from_folder(folder_path):
    jds = {}
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    jds[filename.replace(".txt", "").replace("_", " ").title()] = f.read()
    return jds

# --- Candidate Resume Screener Page ---
def candidate_screener_page():
    st.header("📄 AI Resume Matcher", divider='rainbow')
    st.info("Upload your resume and select a job description (JD) or paste your own to see how well you match!")

    # Load available JDs
    available_jds = load_jds_from_folder(JD_FOLDER)
    jd_options = ["Paste Job Description"] + sorted(list(available_jds.keys()))

    # Use st.selectbox for JD selection
    jd_selection_method = st.selectbox(
        "Choose how to provide the Job Description:",
        jd_options,
        key="jd_selection_method_dropdown"
    )

    jd_text = ""
    selected_jd_name = "User Provided JD Text" # Default for pasted JD

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Job Description")
        if jd_selection_method == "Paste Job Description":
            jd_text = st.text_area(
                "Paste the Job Description (JD) here:",
                height=300,
                placeholder="e.g., We are looking for a Software Engineer with Python and AWS experience...",
                key="candidate_jd_input_paste"
            )
        elif jd_selection_method in available_jds:
            selected_jd_name = jd_selection_method
            jd_text = available_jds[jd_selection_method]
            with st.expander(f"View content of '{selected_jd_name}'"):
                st.text_area(
                    f"Content of '{selected_jd_name}':",
                    value=jd_text,
                    height=250,
                    disabled=True, # Make it read-only
                    key="candidate_jd_input_selected"
                )
        else:
            st.warning("No JDs found in the 'data' folder. Please add some .txt files or paste a JD.")
    
    with col2:
        st.subheader("⬆️ Upload Your Resume")
        # File uploader for single resume
        uploaded_resume = st.file_uploader(
            "Upload your resume here (PDF, DOCX, TXT):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            key="candidate_resume_uploader"
        )
        if uploaded_resume:
            st.success(f"Resume '{uploaded_resume.name}' uploaded successfully!")
        else:
            st.info("Please upload your resume to proceed.")

    st.markdown("---")
    
    # Centering the button using columns
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        if st.button("🚀 Analyze My Resume", key="analyze_candidate_resume_button", help="Click to get your resume match score! 👇"):
            if not jd_text:
                st.error("Please provide a Job Description (paste or select).")
            elif not uploaded_resume:
                st.error("Please upload your resume.")
            else:
                # Set default criteria for candidate app (can be made configurable later)
                default_max_experience = 999 # Effectively no max experience for candidate self-assessment
                default_high_priority_skills = [] # Not exposed to candidate
                default_medium_priority_skills = [] # Not exposed to candidate

                with st.spinner("Analyzing your resume... This may take a moment."):
                    # Read file content once
                    file_bytes = uploaded_resume.read()

                    # Process the single resume using the adapted function from screener_logic
                    screening_result = process_single_resume_for_candidate_app(
                        file_name=uploaded_resume.name,
                        file_bytes=file_bytes,
                        file_type=uploaded_resume.type,
                        jd_text=jd_text,
                        jd_name_for_results=selected_jd_name,
                        high_priority_skills=default_high_priority_skills,
                        medium_priority_skills=default_medium_priority_skills,
                        max_experience=default_max_experience,
                        _global_sentence_model=global_sentence_model,
                        _global_ml_model=global_ml_model
                    )
                    
                    # Check for critical errors from processing
                    if screening_result.get("Tag") == "❌ Text Extraction Error" or screening_result.get("Tag") == "❌ Critical Processing Error":
                        st.error(f"Failed to process resume: {screening_result.get('AI Suggestion', 'Unknown error during processing.')}")
                        return # Stop execution if there's a critical error

                    # Save the result to local history
                    if st.session_state.get('candidate_username'):
                        save_candidate_screening_result_local(st.session_state.candidate_username, screening_result)
                        # Update local leaderboard
                        update_top_candidates_leaderboard_local(
                            screening_result.get('Candidate Name', 'Anonymous'),
                            screening_result.get('Score (%)', 0),
                            st.session_state.candidate_username
                        )

                st.subheader("✨ Your Resume Analysis Results:")

                col_score, col_exp, col_cgpa = st.columns(3)
                col_score.metric("Your Match Score", f"{screening_result['Score (%)']:.2f}%", help="How well your resume matches the job description.")
                col_exp.metric("Years Experience", f"{screening_result['Years Experience']:.1f} yrs", help="Years of experience detected.")
                cgpa_display = f"{screening_result['CGPA (4.0 Scale)']:.2f}" if pd.notna(screening_result['CGPA (4.0 Scale)']) else "N/A"
                col_cgpa.metric("CGPA (4.0 Scale)", cgpa_display, help="Your CGPA normalized to a 4.0 scale.")

                st.markdown("---")

                st.subheader("Detailed Breakdown")
                st.write(f"**Candidate Name:** {screening_result.get('Candidate Name', 'N/A')}")
                st.write(f"**Email:** {screening_result.get('Email', 'N/A')}")
                st.write(f"**Phone Number:** {screening_result.get('Phone Number', 'N/A')}")
                st.write(f"**Location:** {screening_result.get('Location', 'N/A')}")
                st.write(f"**Languages:** {screening_result.get('Languages', 'N/A')}")
                st.write(f"**Semantic Similarity (JD vs. Resume):** {screening_result.get('Semantic Similarity', 'N/A'):.2f}")
                st.write(f"**JD Used:** {screening_result.get('JD Used', 'N/A')}")

                with st.expander("View Matched Keywords"):
                    st.write(screening_result.get('Matched Keywords', 'None'))
                with st.expander("View Missing Skills"):
                    st.write(screening_result.get('Missing Skills', 'None'))
                with st.expander("AI Suggestion"):
                    st.write(screening_result.get('AI Suggestion', 'No specific suggestions.'))
                with st.expander("Detailed HR Assessment"):
                    st.write(screening_result.get('Detailed HR Assessment', 'No detailed assessment.'))

                st.success("Analysis complete!")

                # --- Certificate Generation and Download/Email ---
                if screening_result.get('Score (%)', 0) >= 80:
                    st.markdown("---")
                    st.subheader("🏆 Congratulations! You've earned a Certificate!")
                    
                    # Pass the full screening_result dict to generate_certificate_html
                    certificate_html_content = generate_certificate_html(candidate_data=screening_result)
                    st.session_state['certificate_html_content'] = certificate_html_content # Store for preview

                    # Generate PDF content
                    certificate_pdf_content = generate_certificate_pdf(certificate_html_content)

                    col_cert_email, col_cert_dl = st.columns(2)
                    with col_cert_email:
                        # Modified email sending logic with st.status
                        if st.button("📧 Send Certificate to Email", key="send_cert_email_button"):
                            if not screening_result['Email'] or screening_result['Email'] == "Not Found":
                                st.error("Cannot send certificate: Your email address was not found in the resume. Please ensure it's clearly visible.")
                            elif not st.secrets.get("GMAIL_ADDRESS") or not st.secrets.get("GMAIL_APP_PASSWORD"):
                                st.error("Email sending is not configured. Please ask the app administrator to set up Gmail App Passwords in Streamlit secrets.")
                                st.info("For administrators: Refer to Streamlit documentation on `st.secrets` for setting `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD`.")
                            else:
                                with st.status("Sending certificate email...", expanded=True) as status_box:
                                    status_box.write("Attempting to send email...")
                                    email_send_result = send_certificate_email(
                                        recipient_email=screening_result['Email'],
                                        candidate_name=screening_result['Candidate Name'],
                                        score=screening_result['Score (%)'],
                                        certificate_id=screening_result['Certificate ID'],
                                        certificate_pdf_content=certificate_pdf_content,
                                        gmail_address=st.secrets.get("GMAIL_ADDRESS"),
                                        gmail_app_password=st.secrets.get("GMAIL_APP_PASSWORD")
                                    )
                                    if email_send_result["success"]:
                                        status_box.update(label="Email sent!", state="complete", expanded=False)
                                        st.success(email_send_result["message"])
                                    else:
                                        status_box.update(label="Email failed!", state="error", expanded=True)
                                        st.error(email_send_result["message"])
                    with col_cert_dl:
                        if certificate_pdf_content:
                            st.download_button(
                                label="⬇️ Download Certificate (PDF)",
                                data=certificate_pdf_content,
                                file_name=f"ScreenerPro_Certificate_{screening_result['Candidate Name'].replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key="download_cert_pdf_button"
                            )
                        else:
                            st.warning("PDF generation failed, cannot provide download.")
                    
                    st.markdown("---")
                    st.subheader("Share Your Success!")
                    st.write("Did you make it to the top? Share your achievement!")
                    
                    # Share on X/Twitter using st.link_button
                    x_share_text = urllib.parse.quote(
                        f"I just scored {screening_result.get('Score (%)', 0):.2f}% on the ScreenerPro AI Resume Match! 🎉 Check out my certificate and try it yourself: {APP_BASE_URL} #ScreenerPro #ResumeMatch #JobSearch"
                    )
                    x_share_url = f"https://twitter.com/intent/tweet?text={x_share_text}"
                    st.link_button("Share on X/Twitter", url=x_share_url, help="Share your score on X/Twitter")
                    
                    # LinkedIn share using st.link_button
                    linkedin_share_text = urllib.parse.quote(
                        f"I just scored {screening_result.get('Score (%)', 0):.2f}% on the ScreenerPro AI Resume Match! 🎉 Check out my certificate and try it yourself: {APP_BASE_URL} #ScreenerPro #ResumeMatch #JobSearch"
                    )
                    linkedin_share_url = f"https://www.linkedin.com/shareArticle?mini=true&url={urllib.parse.quote(APP_BASE_URL)}&title={urllib.parse.quote('ScreenerPro AI Match Certificate')}&summary={linkedin_share_text}"
                    st.link_button("Share on LinkedIn", url=linkedin_share_url, help="Share your score on LinkedIn")

                else:
                    st.info(f"Your score of {screening_result.get('Score (%)', 0):.2f}% does not meet the 80% threshold for a ScreenerPro Certificate at this time. Keep improving!")
                
                # --- Suggestions for Improvement ---
                st.markdown("---")
                st.subheader("💡 Suggestions for Improvement")
                missing_skills = screening_result.get('Missing Skills', [])
                if missing_skills and missing_skills != "None": # Check for both empty list and "None" string
                    # Convert string back to list if it was stored as string
                    if isinstance(missing_skills, str):
                        missing_skills = [s.strip() for s in missing_skills.split(',') if s.strip()]

                    if missing_skills: # Re-check if list is not empty after conversion
                        st.write("Based on the job description, consider improving in these areas:")
                        for skill in missing_skills:
                            st.markdown(f"- **{skill.title()}**: Look for courses on Udemy or Coursera. For example, search for '{skill} course Udemy'.")
                            # Placeholder for actual affiliate links
                            udemy_link = f"https://www.udemy.com/courses/search/?q={urllib.parse.quote(skill)}"
                            coursera_link = f"https://www.coursera.org/search?query={urllib.parse.quote(skill)}"
                            st.markdown(f"  [Find on Udemy]({udemy_link}) | [Find on Coursera]({coursera_link})")
                    else:
                        st.info("Great job! No significant missing skills detected for this JD. Keep honing your expertise!")
                else:
                    st.info("Great job! No significant missing skills detected for this JD. Keep honing your expertise!")
                
                st.write("---")
                st.subheader("🚀 General Career Tips:")
                st.markdown("""
                - **Tailor your resume:** Always customize your resume for each job description.
                - **Quantify achievements:** Use numbers and metrics to highlight your impact.
                - **Network:** Connect with professionals in your field.
                - **Practice interviews:** Prepare for common interview questions.
                - **Continuous learning:** The job market evolves; keep your skills updated!
                """)
    
# --- Main Candidate App Logic ---

# --- Dark Mode Toggle (simplified for candidate app) ---
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_candidate")

# --- Load and Inject CSS ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Error: {file_name} not found. Please ensure style.css is in the same directory.")
        return ""

css_content = load_css(STYLE_CSS_FILE)
# Embed CSS directly into the HTML for robust styling
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>{css_content}</style>
<script>
    // JavaScript to apply dark/light mode class to body based on Streamlit toggle
    const darkModeToggle = window.parent.document.querySelector('[data-testid="stSidebar"] [data-testid="stCheckbox"] input');
    const body = window.parent.document.body;

    function applyBodyClass() {{
        if (darkModeToggle && body) {{
            if (darkModeToggle.checked) {{
                body.classList.add('dark-mode');
                body.classList.remove('light-mode');
            }} else {{
                body.classList.add('light-mode');
                body.classList.remove('dark-mode');
            }}
        }}
    }}

    // Initial application
    applyBodyClass();

    // Listen for changes
    if (darkModeToggle) {{
        darkModeToggle.addEventListener('change', applyBodyClass);
    }}
</script>
""", unsafe_allow_html=True)


st.sidebar.title("Candidate Screener")

authenticated = candidate_login_section()

if authenticated:
    st.sidebar.markdown(f"Hello, **{st.session_state.candidate_username}**!")
    
    # Navigation for candidate app (removed "My Dashboard")
    candidate_nav_options = ["🏠 Home (Screener)", "⭐ Top Candidates", "🤝 Refer a Friend"]
    candidate_tab = st.sidebar.radio("Navigate", candidate_nav_options)

    if st.sidebar.button("🚪 Logout", key="candidate_logout_button"):
        st.session_state.candidate_authenticated = False
        st.session_state.pop('candidate_username', None)
        st.success("Logged out successfully.")
        st.rerun()
    
    if candidate_tab == "🏠 Home (Screener)":
        candidate_screener_page()
    elif candidate_tab == "⭐ Top Candidates":
        st.header("⭐ Top Candidates This Week", divider='rainbow')
        
        # Add a logo at the top of the "Top Candidates" page
        st.image("https://placehold.co/150x50/00cec9/ffffff?text=ScreenerPro+Logo", caption="ScreenerPro", width=150, use_column_width=False, output_format="PNG", clamp=True, channels="RGB", format="PNG", class_name="screenerpro-logo")
        
        st.info("See who's acing their resume matches!")
        top_candidates_data = load_top_candidates_leaderboard_local()
        if top_candidates_data:
            top_candidates_df = pd.DataFrame(top_candidates_data)
            # Ensure numeric types
            if 'Score (%)' in top_candidates_df.columns:
                top_candidates_df['Score (%)'] = pd.to_numeric(top_candidates_df['Score (%)'], errors='coerce')
            top_candidates_df = top_candidates_df.sort_values(by='Score (%)', ascending=False).head(10)

            st.dataframe(top_candidates_df[['Candidate Name', 'Score (%)', 'Timestamp']].rename(columns={'Score (%)': 'Score'}), use_container_width=True)
            
            # Add a graph for score distribution
            st.markdown("---")
            st.subheader("Score Distribution Among Top Candidates")
            if not top_candidates_df.empty:
                import plotly.express as px # Import here to avoid circular dependency if not needed elsewhere
                fig = px.histogram(top_candidates_df, x="Score (%)", nbins=10,
                                   title="Distribution of Top Candidate Scores",
                                   labels={"Score (%)": "Candidate Score (%)", "count": "Number of Candidates"},
                                   color_discrete_sequence=[px.colors.qualitative.Pastel[0]])
                # Adjust plot background and font color for dark mode
                fig.update_layout(xaxis_title="Score (%)", yaxis_title="Number of Candidates",
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                  font_color=("#E0E0E0" if dark_mode else "#333333"),
                                  xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                                  yaxis=dict(gridcolor='rgba(128,128,128,0.2)'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to display score distribution yet.")


            st.markdown("---")
            st.subheader("Share Your Success!")
            st.write("Did you make it to the top? Share your achievement!")
            
            # Share on X/Twitter using st.link_button
            x_share_text = urllib.parse.quote(
                f"I just checked my resume match on ScreenerPro and it's awesome! Check out the top candidates here: {APP_BASE_URL} #ScreenerPro #JobSearch #ResumeTips"
            )
            x_share_url = f"https://twitter.com/intent/tweet?text={x_share_text}"
            st.link_button("Share on X/Twitter", url=x_share_url, help="Share this page on X/Twitter")
            
            # LinkedIn share (already defined in certificate, but can be generic here)
            linkedin_share_text_generic = urllib.parse.quote(
                f"Check out the top candidates on ScreenerPro's AI Resume Match platform! Find your fit: {APP_BASE_URL} #ScreenerPro #ResumeMatch #JobSearch"
            )
            linkedin_share_url_generic = f"https://www.linkedin.com/shareArticle?mini=true&url={urllib.parse.quote(APP_BASE_URL)}&title={urllib.parse.quote('ScreenerPro Top Candidates')}&summary={linkedin_share_text_generic}"
            st.link_button("Share on LinkedIn", url=linkedin_share_url_generic, help="Share this page on LinkedIn")

        else:
            st.info("No top candidates to display yet. Be the first to get a high score!")

        st.markdown("---")
        st.subheader("For Recruiters and HR Professionals")
        st.write("Are you an HR professional looking for advanced candidate screening tools?")
        st.link_button("Go to ScreenerPro for HR", url=HR_APP_URL, help="Navigate to the HR ScreenerPro application")


    elif candidate_tab == "🤝 Refer a Friend":
        st.header("🤝 Refer a Friend", divider='rainbow')
        st.info("Help your friends find their perfect job match and get rewarded!")
        st.markdown("""
        Invite 3 friends to sign up and use ScreenerPro, and you'll unlock a **Premium Scan** feature!
        
        **How it works:**
        1. Share your unique referral link below.
        2. Your friends sign up and analyze their resume.
        3. Once 3 friends complete their first scan, we'll notify you about your premium scan!
        """)
        
        # Mock referral link
        referral_link = f"{APP_BASE_URL}?ref={st.session_state.get('candidate_username', 'your_id')}"
        st.code(referral_link)
        st.button("Copy Referral Link", on_click=lambda: st.write("Copied! (functionality not active in sandbox)"))
        
        st.markdown("---")
        st.subheader("Your Referral Status:")
        st.info("You have referred **0** friends. Refer **3** to unlock your premium scan!") # Placeholder

else:
    st.info("Please login or register to use the Candidate Screener.")

