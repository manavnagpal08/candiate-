import streamlit as st
import json
import os
import re
import pandas as pd
import io # For handling resume file content
import bcrypt # For secure password hashing
from datetime import datetime
import base64 # For certificate image embedding
import urllib.parse # For URL encoding for social shares
import collections # For defaultdict in mock screening

# --- Local File Paths ---
USER_DB_FILE_CANDIDATE = "candidate_users.json" # Separate user database for candidates
CANDIDATE_HISTORY_FILE = "candidate_screening_history.json"
CANDIDATE_LEADERBOARD_FILE = "candidate_leaderboard.json"

# --- Authentication Functions (Local JSON) ---
def load_candidate_users():
    """Loads candidate user data from the JSON file."""
    if not os.path.exists(USER_DB_FILE_CANDIDATE):
        with open(USER_DB_FILE_CANDIDATE, "w") as f:
            json.dump({}, f)
    with open(USER_DB_FILE_CANDIDATE, "r") as f:
        users = json.load(f)
        # Ensure each user has a 'status' key for backward compatibility
        for username, data in users.items():
            if isinstance(data, str): # Old format: "username": "hashed_password"
                users[username] = {"password": data, "status": "active"}
            if "status" not in users[username]:
                users[username]["status"] = "active"
        return users

def save_candidate_users(users):
    """Saves candidate user data to the JSON file."""
    with open(USER_DB_FILE_CANDIDATE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    """Checks a password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def is_valid_email(email):
    """Basic validation for email format."""
    return re.match(r"[^@]+@[^@]+\.\w+", email)

def candidate_register_section():
    """Candidate self-registration form using local JSON."""
    st.subheader("📝 Create Your Candidate Account")
    with st.form("candidate_registration_form", clear_on_submit=True):
        new_email = st.text_input("Your Email Address (Username)", key="new_email_cand_reg")
        new_password = st.text_input("Choose Password", type="password", key="new_password_cand_reg")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_cand_reg")
        register_button = st.form_submit_button("Register Account")

        if register_button:
            if not new_email or not new_password or not confirm_password:
                st.error("Please fill in all fields.")
            elif not is_valid_email(new_email):
                st.error("Please enter a valid email address for the username.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users = load_candidate_users()
                if new_email in users:
                    st.error("Email already registered. Please choose a different one or log in.")
                else:
                    users[new_email] = {
                        "password": hash_password(new_password),
                        "status": "active"
                    }
                    save_candidate_users(users)
                    st.success("✅ Registration successful! You can now switch to the 'Login' option.")
                    st.session_state.active_candidate_tab_selection = "Login" # To switch tab after successful registration

def candidate_login_section():
    """Handles candidate login and registration tab switching using local JSON."""
    if "candidate_authenticated" not in st.session_state:
        st.session_state.candidate_authenticated = False
    if "candidate_username" not in st.session_state:
        st.session_state.candidate_username = None

    if st.session_state.get("candidate_authenticated", False):
        return True # Already logged in

    if "active_candidate_tab_selection" not in st.session_state:
        st.session_state.active_candidate_tab_selection = "Login"

    tabs = st.tabs(["Login", "Register"])

    with tabs[0]: # Login tab
        st.subheader("🔐 Candidate Login")
        with st.form("candidate_login_form", clear_on_submit=False):
            email = st.text_input("Your Email Address", key="email_cand_login")
            password = st.text_input("Password", type="password", key="password_cand_login")
            submitted = st.form_submit_button("Login")

            if submitted:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    users = load_candidate_users()
                    if email not in users:
                        st.error("❌ Invalid email or password. Please register if you don't have an account.")
                    else:
                        user_data = users[email]
                        if user_data["status"] == "disabled":
                            st.error("❌ Your account has been disabled. Please contact support.")
                        elif check_password(password, user_data["password"]):
                            st.session_state.candidate_authenticated = True
                            st.session_state.candidate_username = email
                            st.success("✅ Login successful! Redirecting to Screener...")
                            st.rerun()
                            return True
                        else:
                            st.error("❌ Invalid email or password.")
                            return False
    with tabs[1]: # Register tab
        candidate_register_section()

    return False # Not authenticated yet

# --- Local JSON Data Persistence for Candidate History ---
def load_candidate_screening_history_local():
    """Loads all screening results for a candidate from local JSON."""
    if not os.path.exists(CANDIDATE_HISTORY_FILE):
        return []
    with open(CANDIDATE_HISTORY_FILE, "r") as f:
        return json.load(f)

def save_candidate_screening_history_local(history):
    """Saves candidate screening history to local JSON."""
    with open(CANDIDATE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def save_candidate_screening_result_local(username, result_data):
    """Saves a single candidate screening result to local JSON."""
    history = load_candidate_screening_history_local()
    
    # Add username and timestamp
    result_data['user_email'] = username
    result_data['timestamp'] = datetime.now().isoformat()

    # Ensure lists are converted to comma-separated strings for storage if needed
    for key in ['Matched Keywords', 'Missing Skills', 'Languages']:
        if isinstance(result_data.get(key), list):
            result_data[key] = ", ".join(result_data[key])
    
    history.append(result_data)
    save_candidate_screening_history_local(history)
    st.toast("✅ Screening result saved to history!")

def load_top_candidates_local():
    """Loads top candidates from local JSON for leaderboard."""
    if not os.path.exists(CANDIDATE_LEADERBOARD_FILE):
        return []
    with open(CANDIDATE_LEADERBOARD_FILE, "r") as f:
        return json.load(f)

def save_top_candidates_local(leaderboard_data):
    """Saves top candidates leaderboard to local JSON."""
    with open(CANDIDATE_LEADERBOARD_FILE, "w") as f:
        json.dump(leaderboard_data, f, indent=4)

def update_top_candidates_leaderboard_local(candidate_name, score, username):
    """Updates the local leaderboard with a candidate's best score."""
    leaderboard = load_top_candidates_local()
    
    # Check if this user already exists in the leaderboard
    found = False
    for entry in leaderboard:
        if entry.get("SourceUser") == username:
            if score > entry.get("Score (%)", 0):
                entry["Candidate Name"] = candidate_name
                entry["Score (%)"] = score
                entry["Timestamp"] = datetime.now().isoformat()
                st.toast("🎉 Leaderboard updated!")
            found = True
            break
    
    if not found:
        leaderboard.append({
            "Candidate Name": candidate_name,
            "Score (%)": score,
            "Timestamp": datetime.now().isoformat(),
            "SourceUser": username
        })
        st.toast("🎉 Leaderboard updated!")
    
    # Sort and keep only top 10
    leaderboard.sort(key=lambda x: x.get("Score (%)", 0), reverse=True)
    save_top_candidates_local(leaderboard[:10])


# --- Mock Resume Screening Logic (to be replaced by actual ML/NLP if available) ---
def mock_resume_screening_logic(resume_text: str, jd_text: str):
    """
    Mocks a resume screening process, returning a score and feedback.
    This is a placeholder. In a real app, this would use NLP/ML models.
    """
    # Simple keyword matching for demonstration
    jd_keywords = set(re.findall(r'\b[a-zA-Z]{2,}\b', jd_text.lower())) # At least 2 chars
    resume_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', resume_text.lower()))

    common_words = jd_keywords.intersection(resume_words)
    
    # Filter out very common, less meaningful words
    stop_words = {"a", "an", "the", "and", "or", "in", "on", "with", "for", "to", "of", "is", "are", "be", "have", "it", "that", "you", "we", "our", "us", "your", "its", "from", "by", "as", "at", "but", "not", "this", "that", "these", "those", "can", "will", "would", "should", "could", "must", "may", "might", "which", "what", "where", "when", "how", "why", "who", "whom", "whose", "here", "there", "then", "than", "more", "less", "much", "many", "some", "any", "all", "no", "yes", "up", "down", "out", "in", "on", "off", "about", "above", "across", "after", "against", "along", "among", "around", "before", "behind", "below", "beneath", "beside", "between", "beyond", "during", "except", "inside", "into", "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past", "through", "to", "toward", "under", "until", "up", "upon", "with", "within", "without", "with", "without", "etc", "experience", "skills", "knowledge", "ability", "proficient", "strong", "excellent", "proven", "demonstrated", "familiarity", "understanding", "work", "years", "management", "development", "design", "implementation", "testing", "solutions", "systems", "data", "analysis", "project", "team", "client", "customer", "business", "technical", "software", "applications", "tools", "platforms", "frameworks", "technologies", "environment", "processes", "requirements", "support", "delivery", "quality", "performance", "optimization", "integration", "architecture", "strategy", "communication", "problem", "solving", "leadership", "collaboration", "innovative", "creative", "detail", "oriented", "flexible", "adaptable", "independent", "results", "driven", "fast", "paced", "dynamic", "complex", "diverse", "global", "enterprise", "level", "scalable", "robust", "secure", "efficient", "effective", "successful", "key", "core", "advanced", "basic", "fundamental", "conceptual", "practical", "hands", "on", "experience", "with", "in", "on", "of", "and", "or", "to", "for", "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "doing", "will", "would", "should", "can", "could", "may", "might", "must", "about", "above", "across", "after", "against", "along", "among", "around", "at", "before", "behind", "below", "beneath", "beside", "between", "beyond", "but", "by", "down", "during", "except", "for", "from", "in", "inside", "into", "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past", "through", "to", "toward", "under", "until", "up", "upon", "with", "within", "without", "my", "your", "he", "she", "it", "they", "we", "you", "him", "her", "them", "us", "me", "i", "this", "that", "these", "those", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "m", "d", "ll", "ve", "re", "just", "don", "t", "s", "won", "t", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"}
    filtered_common_words = [word for word in common_words if word not in stop_words]

    matched_keywords_list = list(filtered_common_words)[:7] # Take top 7 for slightly more detail

    score = 0
    if len(jd_keywords) > 0:
        score = (len(filtered_common_words) / len(jd_keywords)) * 100
        score = min(100, max(0, score)) # Cap score between 0 and 100

    missing_skills_list = list(jd_keywords - resume_words)[:5] # Top 5 missing skills

    # Mock experience and CGPA (more dynamic)
    # Base experience on score, add some randomness
    years_experience = round(score / 15 + (st.session_state.get('candidate_seed', 0) % 3) + 1, 1)
    years_experience = min(20.0, max(0.0, years_experience)) # Cap at 20 years

    cgpa = round(2.5 + (score / 100 * 1.5) + (st.session_state.get('candidate_seed', 0) % 0.5), 2)
    cgpa = min(4.0, max(2.0, cgpa)) # Cap CGPA between 2.0 and 4.0

    # Mock candidate name and email (extract from resume if possible, otherwise generic)
    candidate_name_match = re.search(r"(?:Name|Candidate Name|Full Name):\s*([A-Za-z\s.-]+)", resume_text, re.IGNORECASE)
    candidate_name = candidate_name_match.group(1).strip() if candidate_name_match else "Valued Candidate"

    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", resume_text)
    email = email_match.group(0) if email_match else "N/A"

    phone_match = re.search(r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", resume_text)
    phone = phone_match.group(0) if phone_match else "N/A"

    location_match = re.search(r"(?:Location|City|Address):\s*([A-Za-z\s,.-]+)", resume_text, re.IGNORECASE)
    location = location_match.group(1).strip() if location_match else "N/A"

    languages_match = re.search(r"(?:Languages|Language Skills):\s*([A-Za-z,\s]+)", resume_text, re.IGNORECASE)
    languages = languages_match.group(1).strip() if languages_match else "N/A"

    # Semantic Similarity (mocked - higher for higher scores)
    semantic_similarity = round(score / 100 * 0.8 + (st.session_state.get('candidate_seed', 0) % 0.15), 2)
    semantic_similarity = min(0.95, max(0.05, semantic_similarity))

    # Store a seed for deterministic mocks during a session
    if 'candidate_seed' not in st.session_state:
        st.session_state.candidate_seed = 0
    st.session_state.candidate_seed += 1

    return {
        "Candidate Name": candidate_name,
        "Email": email,
        "Phone Number": phone,
        "Location": location,
        "Languages": languages,
        "Score (%)": round(score, 2),
        "Years Experience": years_experience,
        "CGPA (4.0 Scale)": cgpa,
        "Matched Keywords": matched_keywords_list, # Return as list for easier processing
        "Missing Skills": missing_skills_list,     # Return as list
        "AI Suggestion": "Your resume shows a good foundation. Focus on building projects related to your missing skills.",
        "Semantic Similarity": semantic_similarity,
        "File Name": uploaded_resume.name if 'uploaded_resume' in locals() else "Uploaded Resume",
        "JD Used": "User Provided JD Text"
    }

# --- Certificate Generation ---
def generate_certificate_html(candidate_name, score, date, app_link="https://your-screener-app.streamlit.app/"):
    """Generates an HTML string for a candidate certificate."""
    badge_svg = """
    <svg width="100" height="100" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2Z" fill="#00cec9"/>
    <path d="M12 16L16 10L12 4L8 10L12 16Z" fill="white"/>
    <path d="M12 16L16 10H8L12 16Z" fill="#00b0a8"/>
    <path d="M12 16L12 22" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M12 2L12 8" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """
    
    html_content = f"""
    <div style="font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #f0f2f6 0%, #e0e2e6 100%); padding: 40px; border-radius: 20px; box-shadow: 0 15px 30px rgba(0,0,0,0.15); max-width: 700px; margin: 30px auto; text-align: center; border: 5px solid #00cec9; position: relative; overflow: hidden;">
        <div style="position: absolute; top: -50px; left: -50px; width: 150px; height: 150px; background: #00cec9; border-radius: 50%; opacity: 0.1;"></div>
        <div style="position: absolute; bottom: -50px; right: -50px; width: 150px; height: 150px; background: #00cec9; border-radius: 50%; opacity: 0.1;"></div>

        <h1 style="color: #00cec9; font-size: 2.5em; margin-bottom: 15px; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">CERTIFICATE OF ACHIEVEMENT</h1>
        <p style="font-size: 1.2em; color: #555; margin-bottom: 20px;">This certifies that</p>
        <h2 style="color: #333; font-size: 2.8em; margin-bottom: 25px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">{candidate_name}</h2>
        <p style="font-size: 1.2em; color: #555; margin-bottom: 10px;">has successfully demonstrated a strong match for a job description with a score of</p>
        <p style="font-size: 3.5em; color: #00b0a8; font-weight: 800; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
            {score:.2f}%
            <span style="font-size: 0.5em; vertical-align: super;">AI Match Score</span>
        </p>
        <div style="margin-bottom: 30px;">
            {badge_svg}
        </div>
        <p style="font-size: 1.1em; color: #777; margin-top: 20px;">Issued on: <strong>{date.strftime('%B %d, %Y')}</strong></p>
        <p style="font-size: 0.9em; color: #999; margin-top: 30px;">Powered by ScreenerPro AI</p>
        <a href="{app_link}" style="display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #00cec9; color: white; text-decoration: none; border-radius: 8px; font-weight: 600;">Visit ScreenerPro</a>
    </div>
    """
    return html_content

# --- Candidate Resume Screener Page ---
def candidate_screener_page():
    st.markdown("## 📄 Single Resume Screener for Candidates")
    st.info("Upload your resume and paste a job description to see how well you match!")

    # Input for Job Description
    jd_text = st.text_area(
        "Paste the Job Description (JD) here:",
        height=200,
        placeholder="e.g., We are looking for a Software Engineer with Python and AWS experience...",
        key="candidate_jd_input"
    )

    # File uploader for single resume
    uploaded_resume = st.file_uploader(
        "Upload Your Resume (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        key="candidate_resume_uploader"
    )

    if st.button("Analyze My Resume", key="analyze_candidate_resume_button"):
        if not jd_text:
            st.error("Please paste a Job Description.")
        elif not uploaded_resume:
            st.error("Please upload your resume.")
        else:
            with st.spinner("Analyzing your resume... This may take a moment."):
                resume_content = ""
                file_extension = uploaded_resume.name.split('.')[-1].lower()

                if file_extension == "txt":
                    resume_content = uploaded_resume.read().decode("utf-8")
                elif file_extension == "pdf":
                    try:
                        import PyPDF2
                        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_resume.read()))
                        for page in reader.pages:
                            resume_content += page.extract_text() + "\n"
                    except ImportError:
                        st.warning("PyPDF2 not installed. Cannot extract text from PDF. Please install it (`pip install PyPDF2`) or upload a .txt/.docx file.")
                        return
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}. Please try a different format or ensure the PDF is text-searchable.")
                        return
                elif file_extension == "docx":
                    try:
                        from docx import Document
                        document = Document(io.BytesIO(uploaded_resume.read()))
                        for paragraph in document.paragraphs:
                            resume_content += paragraph.text + "\n"
                    except ImportError:
                        st.warning("python-docx not installed. Cannot extract text from DOCX. Please install it (`pip install python-docx`) or upload a .txt/.pdf file.")
                        return
                    except Exception as e:
                        st.error(f"Error reading DOCX: {e}.")
                        return
                else:
                    st.error("Unsupported file type. Please upload PDF, DOCX, or TXT.")
                    return

                if not resume_content.strip():
                    st.error("Could not extract text from your resume. Please ensure it's a valid text-based file.")
                    return

                # Perform mock screening
                screening_result = mock_resume_screening_logic(resume_content, jd_text)
                
                # Save the result to local history
                if st.session_state.get('candidate_username'):
                    save_candidate_screening_result_local(st.session_state.candidate_username, screening_result)
                    # Update local leaderboard
                    update_top_candidates_leaderboard_local(
                        screening_result.get('Candidate Name', 'Anonymous'),
                        screening_result.get('Score (%)', 0),
                        st.session_state.candidate_username
                    )


                st.markdown("### ✨ Your Resume Analysis Results:")

                col_score, col_exp, col_cgpa = st.columns(3)
                col_score.metric("Your Match Score", f"{screening_result['Score (%)']:.2f}%", help="How well your resume matches the job description.")
                col_exp.metric("Years Experience", f"{screening_result['Years Experience']:.1f} yrs", help="Years of experience detected.")
                col_cgpa.metric("CGPA (4.0 Scale)", f"{screening_result['CGPA (4.0 Scale)']:.2f}", help="Your CGPA normalized to a 4.0 scale.")

                st.markdown("---")

                st.subheader("Detailed Breakdown")
                st.write(f"**Candidate Name:** {screening_result.get('Candidate Name', 'N/A')}")
                st.write(f"**Email:** {screening_result.get('Email', 'N/A')}")
                st.write(f"**Phone Number:** {screening_result.get('Phone Number', 'N/A')}")
                st.write(f"**Location:** {screening_result.get('Location', 'N/A')}")
                st.write(f"**Languages:** {', '.join(screening_result.get('Languages', [])) if isinstance(screening_result.get('Languages'), list) else screening_result.get('Languages', 'N/A')}")
                st.write(f"**Matched Keywords:** {', '.join(screening_result.get('Matched Keywords', [])) if isinstance(screening_result.get('Matched Keywords'), list) else screening_result.get('Matched Keywords', 'None')}")
                st.write(f"**Missing Skills:** {', '.join(screening_result.get('Missing Skills', [])) if isinstance(screening_result.get('Missing Skills'), list) else screening_result.get('Missing Skills', 'None')}")
                st.write(f"**AI Suggestion:** {screening_result.get('AI Suggestion', 'No specific suggestions.')}")
                st.write(f"**Semantic Similarity (JD vs. Resume):** {screening_result.get('Semantic Similarity', 'N/A'):.2f}")

                st.success("Analysis complete!")

                # --- Certificate Generation and Download ---
                if screening_result.get('Score (%)', 0) >= 80:
                    st.markdown("---")
                    st.subheader("🏆 Congratulations! You've earned a Certificate!")
                    cert_html = generate_certificate_html(
                        candidate_name=screening_result.get('Candidate Name', st.session_state.candidate_username),
                        score=screening_result.get('Score (%)', 0),
                        date=datetime.now(),
                        app_link="https://your-screener-app.streamlit.app/" # Replace with your deployed app URL
                    )
                    st.components.v1.html(cert_html, height=500, scrolling=True)

                    col_cert_dl, col_cert_li = st.columns(2)
                    with col_cert_dl:
                        st.download_button(
                            label="Download Certificate (HTML)",
                            data=cert_html.encode("utf-8"),
                            file_name=f"{screening_result.get('Candidate Name', 'Certificate')}_Match_Certificate.html",
                            mime="text/html",
                            key="download_certificate_html"
                        )
                    with col_cert_li:
                        linkedin_share_text = urllib.parse.quote(
                            f"I just scored {screening_result.get('Score (%)', 0):.2f}% on the ScreenerPro AI Resume Match! 🎉 Check out my certificate and try it yourself: https://your-screener-app.streamlit.app/ #ScreenerPro #ResumeMatch #JobSearch"
                        )
                        linkedin_share_url = f"https://www.linkedin.com/shareArticle?mini=true&url={urllib.parse.quote('https://your-screener-app.streamlit.app/')}&title={urllib.parse.quote('ScreenerPro AI Match Certificate')}&summary={linkedin_share_text}"
                        st.markdown(f'<a href="{linkedin_share_url}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #0A66C2; color: white; border-radius: 8px; text-decoration: none; font-weight: 600;">Share on LinkedIn</a>', unsafe_allow_html=True)

                # --- Suggestions for Improvement ---
                st.markdown("---")
                st.subheader("💡 Suggestions for Improvement")
                missing_skills = screening_result.get('Missing Skills', [])
                if missing_skills:
                    st.write("Based on the job description, consider improving in these areas:")
                    for skill in missing_skills:
                        st.markdown(f"- **{skill.title()}**: Look for courses on Udemy or Coursera. For example, search for '{skill} course Udemy'.")
                        # Placeholder for actual affiliate links
                        udemy_link = f"https://www.udemy.com/courses/search/?q={urllib.parse.quote(skill)}"
                        coursera_link = f"https://www.coursera.org/search?query={urllib.parse.quote(skill)}"
                        st.markdown(f"  [Find on Udemy]({udemy_link}) | [Find on Coursera]({coursera_link})")
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
    
    # --- Candidate Dashboard (always visible after login) ---
    st.markdown("---")
    st.markdown("## 📊 Your Candidate Dashboard")
    
    # Load history
    if st.session_state.get('candidate_username'):
        # Filter history for the current user
        all_history = load_candidate_screening_history_local()
        user_history = [item for item in all_history if item.get('user_email') == st.session_state.candidate_username]
        
        if user_history:
            history_df = pd.DataFrame(user_history)
            # Ensure numeric types
            for col in ['Score (%)', 'Years Experience', 'CGPA (4.0 Scale)', 'Semantic Similarity']:
                if col in history_df.columns:
                    history_df[col] = pd.to_numeric(history_df[col], errors='coerce')
            # Convert timestamp back to datetime and sort
            if 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], errors='coerce')
            history_df = history_df.sort_values(by='timestamp', ascending=False)

            st.subheader("Your Screening History")
            st.dataframe(history_df[['timestamp', 'JD Used', 'Score (%)', 'Years Experience', 'Matched Keywords']].rename(columns={'Score (%)': 'Score', 'Years Experience': 'Experience'}), use_container_width=True)

            # Basic Score Distribution Chart
            st.markdown("#### Your Score Distribution")
            fig_hist = px.histogram(history_df, x='Score (%)', nbins=10, title='Distribution of Your Scores',
                                    labels={'Score (%)': 'Match Score (%)'}, color_discrete_sequence=['#00cec9'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Latest Score Metric
            latest_score = history_df.iloc[0]['Score (%)']
            st.metric("Latest Score", f"{latest_score:.2f}%")

            # Key skills detected from latest screening
            st.subheader("Key Skills from Latest Scan")
            latest_matched_keywords = history_df.iloc[0].get('Matched Keywords', 'None')
            if latest_matched_keywords and isinstance(latest_matched_keywords, str):
                st.markdown(f"**Matched:** {latest_matched_keywords}")
            else:
                st.info("No matched keywords from your latest scan.")

            latest_missing_skills = history_df.iloc[0].get('Missing Skills', 'None')
            if latest_missing_skills and isinstance(latest_missing_skills, str):
                st.markdown(f"**Missing:** {latest_missing_skills}")
            else:
                st.info("No missing skills from your latest scan.")

        else:
            st.info("No screening history found. Upload your resume to see your dashboard!")
    else:
        st.info("Log in to view your screening history dashboard.")

# --- Main Candidate App Logic ---
st.set_page_config(page_title="Candidate Screener – AI Resume Match", layout="centered", page_icon="📄")

# --- Dark Mode Toggle (simplified for candidate app) ---
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_candidate")

# --- Global Fonts & UI Styling (simplified) ---
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
/* Hide Streamlit menu and footer */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {'#1E1E1E' if dark_mode else '#F0F2F6'};
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stMetric {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,{'0.2' if dark_mode else '0.05'});
}}
.stButton>button {{
    background-color: #00cec9;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}
.stButton>button:hover {{
    background-color: #00b0a8;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}}
h1, h2, h3, h4, h5, h6 {{
    color: {'#00cec9' if dark_mode else '#00cec9'};
    font-weight: 700;
}}
.stTextArea > div > div {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    color: {'#E0E0E0' if dark_mode else '#333333'};
    border-radius: 8px;
}}
.stTextArea > label {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stTextInput > div > div > input {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    color: {'#E0E0E0' if dark_mode else '#333333'};
    border-radius: 8px;
}}
.stTextInput > label {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Candidate Screener")

authenticated = candidate_login_section()

if authenticated:
    st.sidebar.markdown(f"Hello, **{st.session_state.candidate_username}**!")
    
    # Navigation for candidate app
    candidate_nav_options = ["🏠 Home (Screener)", "📊 My Dashboard", "⭐ Top Candidates", "🤝 Refer a Friend"]
    candidate_tab = st.sidebar.radio("Navigate", candidate_nav_options)

    if st.sidebar.button("🚪 Logout", key="candidate_logout_button"):
        st.session_state.candidate_authenticated = False
        st.session_state.pop('candidate_username', None)
        st.success("Logged out successfully.")
        st.rerun()
    
    if candidate_tab == "🏠 Home (Screener)":
        candidate_screener_page()
    elif candidate_tab == "📊 My Dashboard":
        st.markdown("## 📊 Your Personal Dashboard")
        if st.session_state.get('candidate_username'):
            # Filter history for the current user
            all_history = load_candidate_screening_history_local()
            user_history = [item for item in all_history if item.get('user_email') == st.session_state.candidate_username]
            
            if user_history:
                history_df = pd.DataFrame(user_history)
                # Ensure numeric types
                for col in ['Score (%)', 'Years Experience', 'CGPA (4.0 Scale)', 'Semantic Similarity']:
                    if col in history_df.columns:
                        history_df[col] = pd.to_numeric(history_df[col], errors='coerce')
                # Convert timestamp back to datetime and sort
                if 'timestamp' in history_df.columns:
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], errors='coerce')
                history_df = history_df.sort_values(by='timestamp', ascending=False)

                st.subheader("Your Screening History")
                st.dataframe(history_df[['timestamp', 'JD Used', 'Score (%)', 'Years Experience', 'Matched Keywords']].rename(columns={'Score (%)': 'Score', 'Years Experience': 'Experience'}), use_container_width=True)

                st.markdown("#### Your Score Distribution")
                fig_hist = px.histogram(history_df, x='Score (%)', nbins=10, title='Distribution of Your Scores',
                                        labels={'Score (%)': 'Match Score (%)'}, color_discrete_sequence=['#00cec9'])
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Latest Score Metric
                latest_score = history_df.iloc[0]['Score (%)']
                st.metric("Latest Score", f"{latest_score:.2f}%")

                st.subheader("Key Skills from Latest Scan")
                latest_matched_keywords = history_df.iloc[0].get('Matched Keywords', 'None')
                if latest_matched_keywords and isinstance(latest_matched_keywords, str):
                    st.markdown(f"**Matched:** {latest_matched_keywords}")
                else:
                    st.info("No matched keywords from your latest scan.")

                latest_missing_skills = history_df.iloc[0].get('Missing Skills', 'None')
                if latest_missing_skills and isinstance(latest_missing_skills, str):
                    st.markdown(f"**Missing:** {latest_missing_skills}")
                else:
                    st.info("No missing skills from your latest scan.")
            else:
                st.info("No screening history found. Use the 'Home (Screener)' tab to analyze your resume!")
        else:
            st.warning("Please log in to view your dashboard.")

    elif candidate_tab == "⭐ Top Candidates":
        st.markdown("## ⭐ Top Candidates This Week")
        st.info("See who's acing their resume matches!")
        top_candidates_data = load_top_candidates_local()
        if top_candidates_data:
            top_candidates_df = pd.DataFrame(top_candidates_data)
            # Ensure numeric types
            if 'Score (%)' in top_candidates_df.columns:
                top_candidates_df['Score (%)'] = pd.to_numeric(top_candidates_df['Score (%)'], errors='coerce')
            top_candidates_df = top_candidates_df.sort_values(by='Score (%)', ascending=False).head(10) # Re-sort to be safe

            st.dataframe(top_candidates_df[['Candidate Name', 'Score (%)', 'Timestamp']].rename(columns={'Score (%)': 'Score'}), use_container_width=True)
            st.markdown("---")
            st.markdown("### Share Your Success!")
            st.write("Did you make it to the top? Share your achievement!")
            
            # Share on X/Twitter
            x_share_text = urllib.parse.quote(
                f"I just checked my resume match on ScreenerPro and it's awesome! Check out the top candidates here: https://your-screener-app.streamlit.app/ #ScreenerPro #JobSearch #ResumeTips"
            )
            x_share_url = f"https://twitter.com/intent/tweet?text={x_share_text}"
            st.markdown(f'<a href="{x_share_url}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #1DA1F2; color: white; border-radius: 8px; text-decoration: none; font-weight: 600; margin-right: 10px;">Share on X/Twitter</a>', unsafe_allow_html=True)
            
            # LinkedIn share (already defined in certificate, but can be generic here)
            linkedin_share_text_generic = urllib.parse.quote(
                f"Check out the top candidates on ScreenerPro's AI Resume Match platform! Find your fit: https://your-screener-app.streamlit.app/ #ScreenerPro #ResumeMatch #JobSearch"
            )
            linkedin_share_url_generic = f"https://www.linkedin.com/shareArticle?mini=true&url={urllib.parse.quote('https://your-screener-app.streamlit.app/')}&title={urllib.parse.quote('ScreenerPro Top Candidates')}&summary={linkedin_share_text_generic}"
            st.markdown(f'<a href="{linkedin_share_url_generic}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #0A66C2; color: white; border-radius: 8px; text-decoration: none; font-weight: 600;">Share on LinkedIn</a>', unsafe_allow_html=True)

        else:
            st.info("No top candidates to display yet. Be the first to get a high score!")

    elif candidate_tab == "🤝 Refer a Friend":
        st.markdown("## 🤝 Refer a Friend")
        st.info("Help your friends find their perfect job match and get rewarded!")
        st.markdown("""
        Invite 3 friends to sign up and use ScreenerPro, and you'll unlock a **Premium Scan** feature!
        
        **How it works:**
        1. Share your unique referral link below.
        2. Your friends sign up and analyze their resume.
        3. Once 3 friends complete their first scan, we'll notify you about your premium scan!
        """)
        
        # Mock referral link
        referral_link = f"https://your-screener-app.streamlit.app/?ref={st.session_state.get('candidate_username', 'your_id')}"
        st.code(referral_link)
        st.button("Copy Referral Link", on_click=lambda: st.write("Copied! (functionality not active in sandbox)"))
        
        st.markdown("---")
        st.subheader("Your Referral Status:")
        st.info("You have referred **0** friends. Refer **3** to unlock your premium scan!") # Placeholder

else:
    st.info("Please login or register to use the Candidate Screener.")

