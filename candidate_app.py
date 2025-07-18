import streamlit as st
import json
import os
import re
import pandas as pd
import io # For handling resume file content
import bcrypt # For secure password hashing
from datetime import datetime, date # Import date specifically
import base64 # For certificate image embedding
import urllib.parse # For URL encoding for social shares
import collections # For defaultdict in mock screening
import numpy as np # For numerical operations
import joblib # For loading ML models
from sklearn.metrics.pairwise import cosine_similarity # For semantic similarity
from sentence_transformers import SentenceTransformer # For embeddings
import nltk # For stopwords
import pdfplumber # For PDF text extraction
from weasyprint import HTML # For PDF certificate generation
import traceback # For detailed error logging
import plotly.express as px # For dashboard histogram
import uuid # Import uuid for certificate IDs

# CRITICAL: Disable Hugging Face tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global NLTK download check (should run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Local File Paths ---
USER_DB_FILE_CANDIDATE = "candidate_users.json" # Separate user database for candidates
CANDIDATE_HISTORY_FILE = "candidate_screening_history.json"
CANDIDATE_LEADERBOARD_FILE = "candidate_leaderboard.json"
JD_FOLDER = "data" # Folder where JDs are stored

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


# --- Constants from screener.py (adapted for candidate_app.py) ---
MASTER_CITIES = set([
    "Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Kochi", "Coimbatore", "Nagpur", "Bhopal", "Indore", "Gurgaon", "Noida", "Surat", "Visakhapatnam",
    "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Varanasi",
    "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Allahabad", "Ranchi", "Jamshedpur", "Gwalior", "Jabalpur",
    "Vijayawada", "Jodhpur", "Raipur", "Kota", "Guwahati", "Thiruvananthapuram", "Mysuru", "Hubballi-Dharwad",
    "Mangaluru", "Belagavi", "Davangere", "Ballari", "Tumakuru", "Shivamogga", "Bidar", "Hassan", "Gadag-Betageri",
    "Chitradurga", "Udupi", "Kolar", "Mandya", "Chikkamagaluru", "Koppal", "Chamarajanagar", "Yadgir", "Raichur",
    "Kalaburagi", "Bengaluru Rural", "Dakshina Kannada", "Uttara Kannada", "Kodagu", "Chikkaballapur", "Ramanagara",
    "Bagalkot", "Gadag", "Haveri", "Vijayanagara", "Krishnagiri", "Vellore", "Salem", "Erode", "Tiruppur", "Madurai",
    "Tiruchirappalli", "Thanjavur", "Dindigul", "Kanyakumari", "Thoothukudi", "Tirunelveli", "Nagercoil", "Puducherry",
    "Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda", "Bicholim", "Curchorem", "Sanquelim", "Valpoi", "Pernem",
    "Quepem", "Canacona", "Mormugao", "Sanguem", "Dharbandora", "Tiswadi", "Salcete", "Bardez",
    "London", "New York", "Paris", "Berlin", "Tokyo", "Sydney", "Toronto", "Vancouver", "Singapore", "Dubai",
    "San Francisco", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego",
    "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte", "Indianapolis",
    "Seattle", "Denver", "Washington D.C.", "Boston", "Nashville", "El Paso", "Detroit", "Oklahoma City",
    "Portland", "Las Vegas", "Memphis", "Louisville", "Baltimore", "Milwaukee", "Albuquerque", "Tucson",
    "Fresno", "Sacramento", "Mesa", "Atlanta", "Kansas City", "Colorado Springs", "Raleigh", "Miami", "Omaha",
    "Virginia Beach", "Long Beach", "Oakland", "Minneapolis", "Tulsa", "Wichita", "New Orleans", "Cleveland",
    "Tampa", "Honolulu", "Anaheim", "Santa Ana", "St. Louis", "Riverside", "Lexington", "Pittsburgh", "Cincinnati",
    "Anchorage", "Plano", "Newark", "Orlando", "Irvine", "Garland", "Hialeah", "Scottsdale", "North Las Vegas",
    "Chandler", "Laredo", "Chula Vista", "Madison", "Reno", "Buffalo", "Durham", "Rochester", "Winston-Salem",
    "St. Petersburg", "Jersey City", "Toledo", "Lincoln", "Greensboro", "Boise", "Richmond", "Stockton",
    "San Bernardino", "Des Moines", "Modesto", "Fayetteville", "Shreveport", "Akron", "Tacoma", "Aurora",
    "Oxnard", "Fontana", "Montgomery", "Little Rock", "Grand Rapids", "Springfield", "Yonkers", "Augusta",
    "Mobile", "Port St. Lucie", "Denton", "Spokane", "Chattanooga", "Worcester", "Providence", "Fort Lauderdale",
    "Chesapeake", "Fremont", "Baton Rouge", "Santa Clarita", "Birmingham", "Glendale", "Huntsville",
    "Salt Lake City", "Frisco", "McKinney", "Grand Prairie", "Overland Park", "Brownsville", "Killeen",
    "Pasadena", "Olathe", "Dayton", "Savannah", "Fort Collins", "Naples", "Gainesville", "Lakeland", "Sarasota",
    "Daytona Beach", "Melbourne", "Clearwater", "St. Augustine", "Key West", "Fort Myers", "Cape Coral",
    "Coral Springs", "Pompano Beach", "Miami Beach", "West Palm Beach", "Boca Raton", "Fort Pierce",
    "Port Orange", "Kissimmee", "Sanford", "Ocala", "Bradenton", "Palm Bay", "Deltona", "Largo",
    "Deerfield Beach", "Boynton Beach", "Coconut Creek", "Sunrise", "Plantation", "Davie", "Miramar",
    "Hollywood", "Pembroke Pines", "Coral Gables", "Doral", "Aventura", "Sunny Isles Beach", "North Miami",
    "Miami Gardens", "Homestead", "Cutler Bay", "Pinecrest", "Kendall", "Richmond Heights", "West Kendall",
    "East Kendall", "South Miami", "Sweetwater", "Opa-locka", "Florida City", "Golden Glades", "Leisure City",
    "Princeton", "West Perrine", "Naranja", "Goulds", "South Miami Heights", "Country Walk", "The Crossings",
    "Three Lakes", "Richmond West", "Palmetto Bay", "Palmetto Estates", "Perrine", "Cutler Ridge", "Westview",
    "Gladeview", "Brownsville", "Liberty City", "West Little River", "Pinewood", "Ojus", "Ives Estates",
    "Highland Lakes", "Sunny Isles Beach", "Golden Beach", "Bal Harbour", "Surfside", "Bay Harbor Islands",
    "Indian Creek", "North Bay Village", "Biscayne Park", "El Portal", "Miami Shores", "North Miami Beach",
    "Aventura"
])

NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
CUSTOM_STOP_WORDS = set([
    "work", "experience", "years", "year", "months", "month", "day", "days", "project", "projects",
    "team", "teams", "developed", "managed", "led", "created", "implemented", "designed",
    "responsible", "proficient", "knowledge", "ability", "strong", "proven", "demonstrated",
    "solution", "solutions", "system", "systems", "platform", "platforms", "framework", "frameworks",
    "database", "databases", "server", "servers", "cloud", "computing", "machine", "learning",
    "artificial", "intelligence", "api", "apis", "rest", "graphql", "agile", "scrum", "kanban",
    "devops", "ci", "cd", "testing", "qa",
    "security", "network", "networking", "virtualization",
    "containerization", "docker", "kubernetes", "terraform", "ansible", "jenkins", "circleci", "github actions", "azure devops", "mlops",
    "containerization", "docker", "kubernetes", "git", "github", "gitlab", "bitbucket", "jira",
    "confluence", "slack", "microsoft", "google", "amazon", "azure", "oracle", "sap", "crm", "erp",
    "salesforce", "servicenow", "tableau", "powerbi", "qlikview", "excel", "word", "powerpoint",
    "outlook", "visio", "html", "css", "js", "web", "data", "science", "analytics", "engineer",
    "software", "developer", "analyst", "business", "management", "reporting", "analysis", "tools",
    "python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "swift", "kotlin", "r",
    "sql", "nosql", "linux", "unix", "windows", "macos", "ios", "android", "mobile", "desktop",
    "application", "applications", "frontend", "backend", "fullstack", "ui", "ux", "design",
    "architecture", "architect", "engineering", "scientist", "specialist", "consultant",
    "associate", "senior", "junior", "lead", "principal", "director", "manager", "head", "chief",
    "officer", "president", "vice", "executive", "ceo", "cto", "cfo", "coo", "hr", "human",
    "resources", "recruitment", "talent", "acquisition", "onboarding", "training", "development",
    "performance", "compensation", "benefits", "payroll", "compliance", "legal", "finance",
    "accounting", "auditing", "tax", "budgeting", "forecasting", "investments", "marketing",
    "sales", "customer", "service", "support", "operations", "supply", "chain", "logistics",
    "procurement", "manufacturing", "production", "quality", "assurance", "control", "research",
    "innovation", "product", "program", "portfolio", "governance", "risk", "communication",
    "presentation", "negotiation", "problem", "solving", "critical", "thinking", "analytical",
    "creativity", "adaptability", "flexibility", "teamwork", "collaboration", "interpersonal",
    "organizational", "time", "multitasking", "detail", "oriented", "independent", "proactive",
    "self", "starter", "results", "driven", "client", "facing", "stakeholder", "engagement",
    "vendor", "budget", "cost", "reduction", "process", "improvement", "standardization",
    "optimization", "automation", "digital", "transformation", "change", "methodologies",
    "industry", "regulations", "regulatory", "documentation", "technical", "writing",
    "dashboards", "visualizations", "workshops", "feedback", "reviews", "appraisals",
    "offboarding", "employee", "relations", "diversity", "inclusion", "equity", "belonging",
    "corporate", "social", "responsibility", "csr", "sustainability", "environmental", "esg",
    "ethics", "integrity", "professionalism", "confidentiality", "discretion", "accuracy",
    "precision", "efficiency", "effectiveness", "scalability", "robustness", "reliability",
    "vulnerability", "assessment", "penetration", "incident", "response", "disaster",
    "recovery", "continuity", "bcp", "drp", "gdpr", "hipaa", "soc2", "iso", "nist", "pci",
    "dss", "ccpa", "privacy", "protection", "grc", "cybersecurity", "information", "infosec",
    "threat", "intelligence", "soc", "event", "siem", "identity", "access", "iam", "privileged",
    "pam", "multi", "factor", "authentication", "mfa", "single", "sign", "on", "sso",
    "encryption", "decryption", "firewall", "ids", "ips", "vpn", "endpoint", "antivirus",
    "malware", "detection", "forensics", "handling", "assessments", "policies", "procedures",
    "guidelines", "mitre", "att&ck", "modeling", "secure", "lifecycle", "sdlc", "awareness",
    "phishing", "vishing", "smishing", "ransomware", "spyware", "adware", "rootkits",
    "botnets", "trojans", "viruses", "worms", "zero", "day", "exploits", "patches", "patching",
    "updates", "upgrades", "configuration", "ticketing", "crm", "erp", "scm", "hcm", "financial",
    "accounting", "bi", "warehousing", "etl", "extract", "transform", "load", "lineage",
    "master", "mdm", "lakes", "marts", "big", "hadoop", "spark", "kafka", "flink", "mongodb",
    "cassandra", "redis", "elasticsearch", "relational", "mysql", "postgresql", "db2",
    "teradata", "snowflake", "redshift", "synapse", "bigquery", "aurora", "dynamodb",
    "documentdb", "cosmosdb", "graph", "neo4j", "graphdb", "timeseries", "influxdb",
    "timescaledb", "columnar", "vertica", "clickhouse", "vector", "pinecone", "weaviate",
    "milvus", "qdrant", "chroma", "faiss", "annoy", "hnswlib", "scikit", "learn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "catboost", "statsmodels", "numpy", "pandas",
    "matplotlib", "seaborn", "plotly", "bokeh", "dash", "flask", "django", "fastapi", "spring",
    "boot", ".net", "core", "node.js", "express.js", "react", "angular", "vue.js", "svelte",
    "jquery", "bootstrap", "tailwind", "sass", "less", "webpack", "babel", "npm", "yarn",
    "ansible", "terraform", "jenkins", "gitlab", "github", "actions", "codebuild", "codepipeline",
    "codedeploy", "build", "deploy", "run", "lambda", "functions", "serverless", "microservices",
    "gateway", "mesh", "istio", "linkerd", "grpc", "restful", "soap", "message", "queues",
    "rabbitmq", "activemq", "bus", "sqs", "sns", "pubsub", "version", "control", "svn",
    "mercurial", "trello", "asana", "monday.com", "smartsheet", "project", "primavera",
    "zendesk", "freshdesk", "itil", "cobit", "prince2", "pmp", "master", "owner", "lean",
    "six", "sigma", "black", "belt", "green", "yellow", "qms", "9001", "27001", "14001",
    "ohsas", "18001", "sa", "8000", "cmii", "cmi", "cism", "cissp", "ceh", "comptia",
    "security+", "network+", "a+", "linux+", "ccna", "ccnp", "ccie", "certified", "solutions",
    "architect", "developer", "sysops", "administrator", "specialty", "professional", "azure",
    "az-900", "az-104", "az-204", "az-303", "az-304", "az-400", "az-500", "az-700", "az-800",
    "az-801", "dp-900", "dp-100", "dp-203", "ai-900", "ai-102", "da-100", "pl-900", "pl-100",
    "pl-200", "pl-300", "pl-400", "pl-500", "ms-900", "ms-100", "ms-101", "ms-203", "ms-500",
    "ms-700", "ms-720", "ms-740", "ms-600", "sc-900", "sc-200", "sc-300", "sc-400", "md-100",
    "md-101", "mb-200", "mb-210", "mb-220", "mb-230", "mb-240", "mb-260", "mb-300", "mb-310",
    "mb-320", "mb-330", "mb-340", "mb-400", "mb-500", "mb-600", "mb-700", "mb-800", "mb-910",
    "mb-920", "gcp-ace", "gcp-pca", "gcp-pde", "gcp-pse", "gcp-pml", "gcp-psa", "gcp-pcd",
    "gcp-pcn", "gcp-psd", "gcp-pda", "gcp-pci", "gcp-pws", "gcp-pwa", "gcp-pme", "gcp-pmc",
    "gcp-pmd", "gcp-pma", "gcp-pmc", "gcp-pmg", "cisco", "juniper", "red", "hat", "rhcsa",
    "rhce", "vmware", "vcpa", "vcpd", "vcpi", "vcpe", "vcpx", "citrix", "cc-v", "cc-p",
    "cc-e", "cc-m", "cc-s", "cc-x", "palo", "alto", "pcnsa", "pcnse", "fortinet", "fcsa",
    "fcsp", "fcc", "fcnsp", "fct", "fcp", "fcs", "fce", "fcn", "fcnp", "fcnse"
])
STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

SKILL_CATEGORIES = {
    "Programming Languages": ["Python", "Java", "JavaScript", "C++", "C#", "Go", "Ruby", "PHP", "Swift", "Kotlin", "TypeScript", "R", "Bash Scripting", "Shell Scripting"],
    "Web Technologies": ["HTML5", "CSS3", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Express.js", "WebSockets"],
    "Databases": ["SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Cassandra", "Elasticsearch", "Neo4j", "Redis", "BigQuery", "Snowflake", "Redshift", "Aurora", "DynamoDB", "DocumentDB", "CosmosDB"],
    "Cloud Platforms": ["AWS", "Azure", "Google Cloud Platform", "GCP", "Serverless", "AWS Lambda", "Azure Functions", "Google Cloud Functions"],
    "DevOps & MLOps": ["Git", "GitHub", "GitLab", "Bitbucket", "CI/CD", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "CircleCI", "GitHub Actions", "Azure DevOps", "MLOps"],
    "Data Science & ML": ["Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Reinforcement Learning", "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "XGBoost", "LightGBM", "Data Cleaning", "Feature Engineering",
    "Model Evaluation", "Statistical Modeling", "Time Series Analysis", "Predictive Modeling", "Clustering",
    "Classification", "Regression", "Neural Networks", "Convolutional Networks", "Recurrent Networks",
    "Transformers", "LLMs", "Prompt Engineering", "Generative AI", "MLOps", "Data Munging", "A/B Testing",
    "Experiment Design", "Hypothesis Testing", "Bayesian Statistics", "Causal Inference", "Graph Neural Networks"],
    "Data Analytics & BI": ["Data Cleaning", "Feature Engineering", "Model Evaluation", "Statistical Analysis", "Time Series Analysis", "Data Munging", "A/B Testing", "Experiment Design", "Hypothesis Testing", "Bayesian Statistics", "Causal Inference", "Excel (Advanced)", "Tableau", "Power BI", "Looker", "Qlik Sense", "Google Data Studio", "Dax", "M Query", "ETL", "ELT", "Data Warehousing", "Data Lake", "Data Modeling", "Business Intelligence", "Data Visualization", "Dashboarding", "Report Generation", "Google Analytics"],
    "Soft Skills": ["Stakeholder Management", "Risk Management", "Change Management", "Communication Skills", "Public Speaking", "Presentation Skills", "Cross-functional Collaboration",
    "Problem Solving", "Critical Thinking", "Analytical Skills", "Adaptability", "Time Management",
    "Organizational Skills", "Attention to Detail", "Leadership", "Mentorship", "Team Leadership",
    "Decision Making", "Negotiation", "Client Management", "Stakeholder Communication", "Active Listening",
    "Creativity", "Innovation", "Research", "Data Analysis", "Report Writing", "Documentation"],
    "Project Management": ["Agile Methodologies", "Scrum", "Kanban", "Jira", "Trello", "Product Lifecycle", "Sprint Planning", "Project Charter", "Gantt Charts", "MVP", "Backlog Grooming",
    "Program Management", "Portfolio Management", "PMP", "CSM"],
    "Security": ["Cybersecurity", "Information Security", "Risk Assessment", "Compliance", "GDPR", "HIPAA", "ISO 27001", "Penetration Testing", "Vulnerability Management", "Incident Response", "Security Audits", "Forensics", "Threat Intelligence", "SIEM", "Firewall Management", "Endpoint Security", "IAM", "Cryptography", "Network Security", "Application Security", "Cloud Security"],
    "Other Tools & Frameworks": ["Jira", "Confluence", "Swagger", "OpenAPI", "Zendesk", "ServiceNow", "Intercom", "Live Chat", "Ticketing Systems", "HubSpot", "Salesforce Marketing Cloud",
    "QuickBooks", "SAP FICO", "Oracle Financials", "Workday", "Microsoft Dynamics", "NetSuite", "Adobe Creative Suite", "Canva", "Mailchimp", "Hootsuite", "Buffer", "SEMrush", "Ahrefs", "Moz", "Screaming Frog",
    "JMeter", "Postman", "SoapUI", "SVN", "Perforce", "Asana", "Monday.com", "Miro", "Lucidchart", "Visio", "MS Project", "Primavera", "AutoCAD", "SolidWorks", "MATLAB", "LabVIEW", "Simulink", "ANSYS",
    "CATIA", "NX", "Revit", "ArcGIS", "QGIS", "OpenCV", "NLTK", "SpaCy", "Gensim", "Hugging Face Transformers",
    "Docker Compose", "Helm", "Ansible Tower", "SaltStack", "Chef InSpec", "Terraform Cloud", "Vault",
    "Consul", "Nomad", "Prometheus", "Grafana", "Alertmanager", "Loki", "Tempo", "Jaeger", "Zipkin",
    "Fluentd", "Logstash", "Kibana", "Grafana Loki", "Datadog", "New Relic", "AppDynamics", "Dynatrace",
    "Nagios", "Zabbix", "Icinga", "PRTG", "SolarWinds", "Wireshark", "Nmap", "Metasploit", "Burp Suite",
    "OWASP ZAP", "Nessus", "Qualys", "Rapid7", "Tenable", "CrowdStrike", "SentinelOne", "Palo Alto Networks",
    "Fortinet", "Cisco Umbrella", "Okta", "Auth0", "Keycloak", "Ping Identity", "Active Directory",
    "LDAP", "OAuth", "JWT", "OpenID Connect", "SAML", "MFA", "SSO", "PKI", "TLS/SSL", "VPN", "IDS/IPS",
    "DLP", "CASB", "SOAR", "XDR", "EDR", "MDR", "GRC", "ITIL", "Lean Six Sigma", "CFA", "CPA", "SHRM-CP",
    "PHR", "CEH", "OSCP", "CCNA", "CISSP", "CISM", "CompTIA Security+"]
}

MASTER_SKILLS = set([skill for category_list in SKILL_CATEGORIES.values() for skill in category_list])

# IMPORTANT: REPLACE THESE WITH YOUR ACTUAL DEPLOYMENT URLs
APP_BASE_URL = "https://your-screener-app.streamlit.app" # Placeholder
CERTIFICATE_HOSTING_URL = "https://your-certificate-hosting-url.com/certificates" # Placeholder for certificate hosting


# Load ML models once using st.cache_resource
@st.cache_resource
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Ensure ml_screening_model.pkl is created and available
        # For demonstration, we'll create a dummy one if it doesn't exist
        if not os.path.exists("ml_screening_model.pkl"):
            st.warning("ml_screening_model.pkl not found. Creating a dummy model for demonstration. Please replace with your actual trained model.")
            from sklearn.linear_model import LogisticRegression
            dummy_model = LogisticRegression()
            # Create dummy data for the model to "fit" on
            # Embeddings are 384-dimensional for all-MiniLM-L6-v2
            dummy_X = np.random.rand(10, 384 * 2 + 2) # JD_embed + Resume_embed + Exp + Weighted_Keyword_Overlap
            dummy_y = np.random.randint(0, 101, 10) # Scores from 0-100
            dummy_model.fit(dummy_X, dummy_y)
            joblib.dump(dummy_model, "ml_screening_model.pkl")

        ml_model = joblib.load("ml_screening_model.pkl")
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading ML models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory and is a valid joblib file.")
        return None, None

# Load models globally (once per app run)
global_sentence_model, global_ml_model = load_ml_model()


def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip().lower()

def extract_relevant_keywords(text, filter_set):
    cleaned_text = clean_text(text)
    extracted_keywords = set()
    categorized_keywords = collections.defaultdict(list)

    if filter_set:
        sorted_filter_skills = sorted(list(filter_set), key=len, reverse=True)
        
        temp_text = cleaned_text

        for skill_phrase in sorted_filter_skills:
            pattern = r'\b' + re.escape(skill_phrase.lower()) + r'\b'
            
            matches = re.findall(pattern, temp_text)
            if matches:
                extracted_keywords.add(skill_phrase.lower())
                found_category = False
                for category, skills_in_category in SKILL_CATEGORIES.items():
                    if skill_phrase.lower() in [s.lower() for s in skills_in_category]:
                        categorized_keywords[category].append(skill_phrase)
                        found_category = True
                        break
                if not found_category:
                    categorized_keywords["Uncategorized"].append(skill_phrase)

                temp_text = re.sub(pattern, " ", temp_text)
        
        individual_words_remaining = set(re.findall(r'\b\w+\b', temp_text))
        for word in individual_words_remaining:
            if word in filter_set:
                extracted_keywords.add(word)
                found_category = False
                for category, skills_in_category in SKILL_CATEGORIES.items():
                    if word.lower() in [s.lower() for s in skills_in_category]:
                        categorized_keywords[category].append(word)
                        found_category = True
                        break
                if not found_category:
                    categorized_keywords["Uncategorized"].append(word)

    else:
        all_words = set(re.findall(r'\b\w+\b', cleaned_text))
        extracted_keywords = {word for word in all_words if word not in STOP_WORDS}
        for word in extracted_keywords:
            categorized_keywords["General Keywords"].append(word)

    return extracted_keywords, dict(categorized_keywords)

# Modified extract_text_from_file to remove OCR
def extract_text_from_file(file_bytes, file_name, file_type):
    full_text = ""

    if "pdf" in file_type:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                full_text = ''.join(page.extract_text() or '' for page in pdf.pages)
        except Exception as e:
            st.error(f"Error reading PDF file {file_name}: {e}. Please ensure it's a valid, text-based PDF.")
            return f"[ERROR] Failed to extract text from PDF: {e}"
    elif "txt" in file_type:
        try:
            full_text = file_bytes.decode("utf-8")
        except Exception as e:
            st.error(f"Error reading TXT file {file_name}: {e}.")
            return f"[ERROR] Failed to extract text from TXT: {e}"
    else:
        st.error(f"Unsupported file type for {file_name}: {file_type}. Only PDF and TXT are supported without OCR.")
        return f"[ERROR] Unsupported file type: {file_type}. Please upload a PDF or TXT file."

    if not full_text.strip():
        st.warning(f"No readable text extracted from {file_name}. It might be an empty document or a scanned PDF without text layer.")
        return "[ERROR] No readable text extracted from the file."
    
    return full_text


def extract_years_of_experience(text):
    text = text.lower()
    total_months = 0
    
    date_patterns = [
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*(?:to|–|-)\s*(present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        r'(\b\d{4})\s*(?:to|–|-)\s*(present|\b\d{4})'
    ]

    for pattern in date_patterns:
        job_date_ranges = re.findall(pattern, text)
        for start_str, end_str in job_date_ranges:
            start_date = None
            end_date = None

            try:
                start_date = datetime.strptime(start_str.strip(), '%B %Y')
            except ValueError:
                try:
                    start_date = datetime.strptime(start_str.strip(), '%b %Y')
                except ValueError:
                    try:
                        start_date = datetime(int(start_str.strip()), 1, 1)
                    except ValueError:
                        pass

            if start_date is None:
                continue

            if end_str.strip() == 'present':
                end_date = datetime.now()
            else:
                try:
                    end_date = datetime.strptime(end_str.strip(), '%B %Y')
                except ValueError:
                    try:
                        end_date = datetime.strptime(end_str.strip(), '%b %Y')
                    except ValueError:
                        try:
                            end_date = datetime(int(end_str.strip()), 12, 31)
                        except ValueError:
                            pass
            
            if end_date is None:
                continue

            delta_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            total_months += max(delta_months, 0)

    if total_months > 0:
        return round(total_months / 12, 1)
    else:
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

    return 0.0

def extract_email(text):
    text = text.lower()

    text = text.replace("gmaill.com", "gmail.com").replace("gmai.com", "gmail.com")
    text = text.replace("yah00", "yahoo").replace("outiook", "outlook")
    text = text.replace("coim", "com").replace("hotmai", "hotmail")

    text = re.sub(r'[^\w\s@._+-]', ' ', text)

    possible_emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.\w+', text)

    if possible_emails:
        # Prioritize common domains or specific keywords if needed
        for email in possible_emails:
            if "gmail" in email or "outlook" in email or "yahoo" in email:
                return email
        return possible_emails[0]
    
    return None

def extract_phone_number(text):
    # Updated regex to be more robust for various formats
    match = re.search(r'(\+?\d{1,4}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    return match.group(0) if match else None

def extract_location(text):
    found_locations = set()
    text_lower = text.lower()

    sorted_cities = sorted(list(MASTER_CITIES), key=len, reverse=True)

    for city in sorted_cities:
        pattern = r'\b' + re.escape(city.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_locations.add(city)

    if found_locations:
        return ", ".join(sorted(list(found_locations)))
    return "Not Found"

def extract_name(text):
    lines = text.strip().split('\n')
    if not lines:
        return None

    EXCLUDE_NAME_TERMS = {
    "linkedin", "github", "portfolio", "resume", "cv", "profile", "contact", "email", "phone",
    "mobile", "number", "tel", "telephone", "address", "website", "site", "social", "media",
    "url", "link", "blog", "personal", "summary", "about", "objective", "dob", "birth", "age",
    "nationality", "gender", "location", "city", "country", "pin", "zipcode", "state", "whatsapp",
    "skype", "telegram", "handle", "id", "details", "connection", "reach", "network", "www",
    "https", "http", "contactinfo", "connect", "reference", "references","fees"}


    potential_name_lines = []
    for line in lines[:5]: # Check first few lines for name
        line = line.strip()
        line_lower = line.lower()

        if not re.search(r'[@\d\.\-]', line) and \
           len(line.split()) <= 4 and \
           not any(term in line_lower for term in EXCLUDE_NAME_TERMS):
            if line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split())):
                potential_name_lines.append(line)

    if potential_name_lines:
        name = max(potential_name_lines, key=len) # Take the longest potential name
        # Clean up common resume section headers if they accidentally get picked
        name = re.sub(r'summary|education|experience|skills|projects|certifications|profile|contact', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', name).strip() # Remove leading/trailing non-alphanumeric
        if name:
            return name.title()
    return None

def extract_cgpa(text):
    text = text.lower()
    
    matches = re.findall(r'(?:cgpa|gpa|grade point average)\s*[:\s]*(\d+\.\d+)(?:\s*[\/of]{1,4}\s*(\d+\.\d+|\d+))?|(\d+\.\d+)(?:\s*[\/of]{1,4}\s*(\d+\.\d+|\d+))?\s*(?:cgpa|gpa)', text)

    for match in matches:
        if match[0] and match[0].strip():
            raw_cgpa = float(match[0])
            scale = float(match[1]) if match[1] else None
        elif match[2] and match[2].strip():
            raw_cgpa = float(match[2])
            scale = float(match[3]) if match[3] else None
        else:
            continue

        if scale and scale not in [0, 1]: # Avoid division by zero or scale of 1 (which is usually just the value)
            normalized_cgpa = (raw_cgpa / scale) * 4.0
            return round(normalized_cgpa, 2)
        elif raw_cgpa <= 4.0: # Assume it's already on a 4.0 scale
            return round(raw_cgpa, 2)
        elif raw_cgpa <= 10.0: # Assume it's on a 10.0 scale, normalize to 4.0
            return round((raw_cgpa / 10.0) * 4.0, 2)
        
    return None

def extract_education_text(text):
    """
    Extracts a single-line education entry from resume text.
    Returns a clean string like: "B.Tech in CSE, Alliance University, Bangalore – 2028"
    Works with or without 'Expected' in the year.
    """

    text = text.replace('\r', '').replace('\t', ' ')
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    education_section = ''
    capture = False

    # Heuristic to find education section
    for line in lines:
        line_lower = line.lower()
        if any(h in line_lower for h in ['education', 'academic background', 'qualifications']):
            capture = True
            continue
        # Stop capturing if another major section starts
        if capture and any(h in line_lower for h in ['experience', 'skills', 'certifications', 'projects', 'languages']):
            break
        if capture:
            education_section += line + ' '

    education_section = education_section.strip()

    # Try to find a pattern that includes university/college name and a year
    edu_match = re.search(
        r'([A-Za-z0-9.,()&\-\s]+?(university|college|institute|school)[^–\n]{0,50}[–\-—]?\s*(expected\s*)?\d{4})',
        education_section,
        re.IGNORECASE
    )

    if edu_match:
        return edu_match.group(1).strip()

    # Fallback to common degree patterns with a year
    fallback_match = re.search(
        r'([A-Za-z0-9.,()&\-\s]+?(b\.tech|m\.tech|b\.sc|m\.sc|bca|bba|mba|ph\.d)[^–\n]{0,50}\d{4})',
        education_section,
        re.IGNORECASE
    )
    if fallback_match:
        return fallback_match.group(1).strip()

    # Fallback to just the first line of the education section if no specific pattern found
    fallback_line = education_section.split('.')[0].strip()
    return fallback_line if fallback_line else None

def extract_work_history(text):
    work_history_section_matches = re.finditer(r'(?:experience|work history|employment history)\s*(\n|$)', text, re.IGNORECASE)
    work_details = []
    
    start_index = -1
    for match in work_history_section_matches:
        start_index = match.end()
        break

    if start_index != -1:
        sections = ['education', 'skills', 'projects', 'certifications', 'awards', 'publications']
        end_index = len(text)
        for section in sections:
            section_match = re.search(r'\b' + re.escape(section) + r'\b', text[start_index:], re.IGNORECASE)
            if section_match:
                end_index = start_index + section_match.start()
                break
        
        work_text = text[start_index:end_index].strip()
        
        # Split by common patterns indicating a new job entry
        job_blocks = re.split(r'\n(?=[A-Z][a-zA-Z\s,&\.]+(?:\s(?:at|@))?\s*[A-Z][a-zA-Z\s,&\.]*\s*(?:-|\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}))', work_text, flags=re.IGNORECASE)
        
        for block in job_blocks:
            block = block.strip()
            if not block:
                continue
            
            company = None
            title = None
            start_date = None
            end_date = None

            # Extract date range first
            date_range_match = re.search(
                r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{4})\s*[-–]\s*(present|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{4})',
                block, re.IGNORECASE
            )
            if date_range_match:
                start_date = date_range_match.group(1)
                end_date = date_range_match.group(2)
                block = block.replace(date_range_match.group(0), '').strip() # Remove dates from block for other extractions

            lines = block.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue

                # Try to find "Title at Company" pattern
                title_company_match = re.search(r'([A-Z][a-zA-Z\s,\-&.]+)\s+(?:at|@)\s+([A-Z][a-zA-Z\s,\-&.]+)', line)
                if title_company_match:
                    title = title_company_match.group(1).strip()
                    company = title_company_match.group(2).strip()
                    break
                
                # Try to find "Company, Title" pattern
                company_title_match = re.search(r'^([A-Z][a-zA-Z\s,\-&.]+),\s*([A-Z][a-zA-Z\s,\-&.]+)', line)
                if company_title_match:
                    company = company_title_match.group(1).strip()
                    title = company_title_match.group(2).strip()
                    break
                
                # Fallback: if no clear title/company, take the first capitalized phrase as potential org/title
                if not company and not title:
                    potential_org_match = re.search(r'^[A-Z][a-zA-Z\s,\-&.]+', line)
                    if potential_org_match and len(potential_org_match.group(0).split()) > 1:
                        if not company: company = potential_org_match.group(0).strip()
                        elif not title: title = potential_org_match.group(0).strip()
                        break

            if company or title or start_date or end_date:
                work_details.append({
                    "Company": company,
                    "Title": title,
                    "Start Date": start_date,
                    "End Date": end_date
                })
    return work_details

def extract_project_details(text, MASTER_SKILLS):
    """
    Extracts real project entries from resume text.
    Returns a list of dicts: Title, Description, Technologies Used
    """

    project_details = []

    text = text.replace('\r', '').replace('\t', ' ')
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # Step 1: Isolate project section
    project_section_keywords = r'(projects|personal projects|key projects|portfolio|selected projects|major projects|academic projects|relevant projects)'
    project_section_match = re.search(project_section_keywords + r'\s*(\n|$)', text, re.IGNORECASE)

    if not project_section_match:
        # Fallback if no explicit project section found, consider the entire text or a large chunk
        project_text = text[:1000]  # fallback to first 1000 chars
        start_index = 0
    else:
        start_index = project_section_match.end()
        # Define keywords for sections that might follow projects, to know where to stop
        sections = ['education', 'skills', 'experience', 'certifications', 'awards', 'publications', 'interests', 'hobbies']
        end_index = len(text)
        for section in sections:
            match = re.search(r'\b' + re.escape(section) + r'\b', text[start_index:], re.IGNORECASE)
            if match:
                end_index = start_index + match.start()
                break
        project_text = text[start_index:end_index].strip()

    if not project_text:
        return []

    lines = [line.strip() for line in project_text.split('\n') if line.strip()]
    current_project = {"Project Title": None, "Description": [], "Technologies Used": set()}

    forbidden_title_keywords = [
        'skills gained', 'responsibilities', 'reflection', 'summary',
        'achievements', 'capabilities', 'what i learned', 'tools used'
    ]

    for i, line in enumerate(lines):
        line_lower = line.lower()

        # Skip all-uppercase names or headers that are too short to be titles
        if re.match(r'^[A-Z\s]{5,}$', line) and len(line.split()) <= 4:
            continue

        # Check if previous line was a bullet point, which might indicate continuation of previous project
        prev_line_is_bullet = False
        if i > 0 and re.match(r'^[•*-]', lines[i - 1]):
            prev_line_is_bullet = True

        # Criteria for a strong new project title:
        # - Starts with a number or bullet, or the word "project"
        # - Not just a soft skill block or forbidden keyword
        # - Contains a reasonable number of words (3-15)
        # - Not entirely in uppercase (suggests a header, not a title)
        is_title = (
            (re.match(r'^[•*-]?\s*\d+[\).:-]?\s', line) or line.lower().startswith("project")) and
            3 <= len(line.split()) <= 15 and
            not any(kw in line_lower for kw in forbidden_title_keywords) and
            not prev_line_is_bullet and
            not line.isupper()
        )

        is_url = re.match(r'https?://', line_lower) # Check if line is a URL

        # If a new project title is detected or a URL (often project links)
        if is_title or is_url:
            # If there's an existing project being built, save it first
            if current_project["Project Title"] or current_project["Description"]:
                full_desc = "\n".join(current_project["Description"])
                techs, _ = extract_relevant_keywords(full_desc, MASTER_SKILLS)
                current_project["Technologies Used"].update(techs)

                project_details.append({
                    "Project Title": current_project["Project Title"],
                    "Description": full_desc.strip(),
                    "Technologies Used": ", ".join(sorted(current_project["Technologies Used"]))
                })

            # Start a new project entry
            current_project = {"Project Title": line, "Description": [], "Technologies Used": set()}
        else:
            # Otherwise, append the line to the current project's description
            current_project["Description"].append(line)

    # Add the last project after the loop finishes
    if current_project["Project Title"] or current_project["Description"]:
        full_desc = "\n".join(current_project["Description"])
        techs, _ = extract_relevant_keywords(full_desc, MASTER_SKILLS)
        current_project["Technologies Used"].update(techs)

        project_details.append({
            "Project Title": current_project["Project Title"],
            "Description": full_desc.strip(),
            "Technologies Used": ", ".join(sorted(current_project["Technologies Used"]))
        })

    return project_details


def extract_languages(text):
    """
    Extracts known languages from resume text.
    Returns a comma-separated string of detected languages or 'Not Found'.
    """
    languages_list = set()
    cleaned_full_text = clean_text(text)

    # De-duplicated, lowercase language set (expanded from screener.py)
    all_languages = list(set([
        "english", "hindi", "spanish", "french", "german", "mandarin", "japanese", "arabic",
        "russian", "portuguese", "italian", "korean", "bengali", "marathi", "telugu", "tamil",
        "gujarati", "urdu", "kannada", "odia", "malayalam", "punjabi", "assamese", "kashmiri",
        "sindhi", "sanskrit", "dutch", "swedish", "norwegian", "danish", "finnish", "greek",
        "turkish", "hebrew", "thai", "vietnamese", "indonesian", "malay", "filipino", "swahili",
        "farsi", "persian", "polish", "ukrainian", "romanian", "czech", "slovak", "hungarian",
        "chinese", "tagalog", "nepali", "sinhala", "burmese", "khmer", "lao", "pashto", "dari",
        "uzbek", "kazakh", "azerbaijani", "georgian", "armenian", "albanian", "serbian",
        "croatian", "bosnian", "bulgarian", "macedonian", "slovenian", "estonian", "latvian",
        "lithuanian", "icelandic", "irish", "welsh", "gaelic", "maltese", "esperanto", "latin",
        "ancient greek", "modern greek", "yiddish", "romani", "catalan", "galician", "basque",
        "breton", "cornish", "manx", "frisian", "luxembourgish", "sami", "romansh", "sardinian",
        "corsican", "occitan", "provencal", "walloon", "flemish", "afrikaans", "zulu", "xhosa",
        "sesotho", "setswana", "shona", "ndebele", "venda", "tsonga", "swati", "kikuyu",
        "luganda", "kinyarwanda", "kirundi", "lingala", "kongo", "yoruba", "igbo", "hausa",
        "fulani", "twi", "ewe", "ga", "dagbani", "gur", "mossi", "bambara", "senufo", "wolof",
        "mandinka", "susu", "krio", "temne", "limba", "mende", "gola", "vai", "kpelle", "loma",
        "bandi", "bassa", "grebo", "krahn", "dan", "mano", "guerze", "kono", "kisi"
    ]))

    sorted_all_languages = sorted(all_languages, key=len, reverse=True)

    # Step 1: Try to locate a language-specific section
    section_match = re.search(
        r'\b(languages|language skills|linguistic abilities|known languages)\s*[:\-]?\s*\n?',
        cleaned_full_text, re.IGNORECASE
    )

    if section_match:
        start_index = section_match.end()
        # Optional: stop at next known section
        end_index = len(cleaned_full_text)
        stop_words = ['education', 'experience', 'skills', 'certifications', 'awards', 'publications', 'interests', 'hobbies']
        for stop in stop_words:
            m = re.search(r'\b' + stop + r'\b', cleaned_full_text[start_index:], re.IGNORECASE)
            if m:
                end_index = start_index + m.start()
                break

        language_chunk = cleaned_full_text[start_index:end_index]
    else:
        language_chunk = cleaned_full_text

    # Step 2: Match known languages
    for lang in sorted_all_languages:
        # Use word boundaries for exact matches and allow for common suffixes like " (fluent)"
        pattern = r'\b' + re.escape(lang) + r'(?:\s*\(?[a-z\s,-]+\)?)?\b'
        if re.search(pattern, language_chunk, re.IGNORECASE):
            languages_list.add(lang.title()) # Add in title case

    return ", ".join(sorted(languages_list)) if languages_list else "Not Found"


def format_work_history(work_list):
    if not work_list:
        return "Not Found"
    formatted_entries = []
    for entry in work_list:
        parts = []
        if entry.get("Title"):
            parts.append(entry["Title"])
        if entry.get("Company"):
            parts.append(f"at {entry['Company']}")
        if entry.get("Start Date") and entry.get("End Date"):
            parts.append(f"({entry['Start Date']} - {entry['End Date']})")
        elif entry.get("Start Date"):
            parts.append(f"(Since {entry['Start Date']})")
        formatted_entries.append(" ".join(parts).strip())
    return "; ".join(formatted_entries) if formatted_entries else "Not Found"

def format_project_details(proj_list):
    if not proj_list:
        return "Not Found"
    formatted_entries = []
    for entry in proj_list:
        parts = []
        if entry.get("Project Title"):
            parts.append(f"**{entry['Project Title']}**")
        if entry.get("Technologies Used"):
            parts.append(f"({entry['Technologies Used']})")
        if entry.get("Description") and entry["Description"].strip():
            desc_snippet = entry["Description"].split('\n')[0][:50] + "..." if len(entry["Description"]) > 50 else entry["Description"]
            parts.append(f'"{desc_snippet}"')
        formatted_entries.append(" ".join(parts).strip())
    return "; ".join(formatted_entries) if formatted_entries else "Not Found"

@st.cache_data(show_spinner="Generating concise AI Suggestion...")
def generate_concise_ai_suggestion(candidate_name, score, years_exp, semantic_similarity, cgpa):
    overall_fit_description = ""
    review_focus_text = ""
    key_strength_hint = ""

    high_score = 85
    moderate_score = 65
    high_exp = 4
    moderate_exp = 2
    high_sem_sim = 0.75
    moderate_sem_sim = 0.4
    high_cgpa = 3.5
    moderate_cgpa = 3.0

    if score >= high_score and years_exp >= high_exp and semantic_similarity >= high_sem_sim:
        overall_fit_description = "High alignment."
        key_strength_hint = "Strong technical and experience match, quick integration expected."
        review_focus_text = "Cultural fit, project contributions."
    elif score >= moderate_score and years_exp >= moderate_exp and semantic_similarity >= moderate_sem_sim:
        overall_fit_description = "Moderate fit."
        key_strength_hint = "Good foundational skills, potential for growth."
        review_focus_text = "Depth of experience, skill application, learning agility."
    else:
        overall_fit_description = "Limited alignment."
        key_strength_hint = "May require significant development or a different role."
        review_focus_text = "Foundational skills, transferable experience, long-term potential."

    cgpa_note = ""
    if cgpa is not None:
        if cgpa >= high_cgpa:
            cgpa_note = "Excellent academic record. "
        elif cgpa >= moderate_cgpa:
            cgpa_note = "Solid academic background. "
        else:
            cgpa_note = "Academic record may need review. "
    else:
        cgpa_note = "CGPA not found. "

    summary_text = f"**Fit:** {overall_fit_description} **Strengths:** {cgpa_note}{key_strength_hint} **Focus:** {review_focus_text}"
    return summary_text

@st.cache_data(show_spinner="Generating detailed HR Assessment...")
def generate_detailed_hr_assessment(candidate_name, score, years_exp, semantic_similarity, cgpa, jd_text, resume_text, matched_keywords, missing_skills, max_exp_cutoff):
    assessment_parts = []
    overall_assessment_title = ""
    next_steps_focus = ""

    matched_kws_str = ", ".join(matched_keywords) if isinstance(matched_keywords, list) else matched_keywords
    missing_skills_str = ", ".join(missing_skills) if isinstance(missing_skills, list) else missing_skills

    high_score = 90
    strong_score = 80
    promising_score = 60
    high_exp = 5
    strong_exp = 3
    promising_exp = 1
    high_sem_sim = 0.85
    strong_sem_sim = 0.7
    promising_sem_sim = 0.35
    high_cgpa = 3.5
    strong_cgpa = 3.0
    promising_cgpa = 2.5

    if score >= high_score and years_exp >= high_exp and years_exp <= max_exp_cutoff and semantic_similarity >= high_sem_sim and (cgpa is None or cgpa >= high_cgpa):
        overall_assessment_title = "Exceptional Candidate: Highly Aligned with Strategic Needs"
        assessment_parts.append(f"**{candidate_name}** presents an **exceptional profile** with a high score of {score:.2f}% and {years_exp:.1f} years of experience. This demonstrates a profound alignment with the job description's core requirements, further evidenced by a strong semantic similarity of {semantic_similarity:.2f}.")
        if cgpa is not None:
            assessment_parts.append(f"Their academic record, with a CGPA of {cgpa:.2f} (normalized to 4.0 scale), further solidifies their strong foundational knowledge.")
        assessment_parts.append(f"**Key Strengths:** This candidate possesses a robust skill set directly matching critical keywords in the JD, including: *{matched_kws_str if matched_kws_str else 'No specific keywords listed, but overall strong match'}*. Their extensive experience indicates a capacity for leadership and handling complex challenges, suggesting immediate productivity and minimal ramp-up time. They are poised to make significant contributions from day one.")
        assessment_parts.append("The resume highlights a clear career progression and a history of successful project delivery, often exceeding expectations. Their qualifications exceed expectations, making them a top-tier applicant for this role.")
        assessment_parts.append("This individual's profile suggests they are not only capable of fulfilling the role's duties but also have the potential to mentor others, drive innovation, and take on strategic initiatives within the team. Their background indicates a strong fit for a high-impact position.")
        next_steps_focus = "The next steps should focus on assessing cultural integration, exploring leadership potential, and delving into strategic contributions during the interview. Prepare for a deep dive into their most challenging projects, how they navigated complex scenarios, and their long-term vision. Consider fast-tracking this candidate through the interview process and potentially involving senior leadership early on."
        assessment_parts.append(f"**Action:** Strongly recommend for immediate interview. Prioritize for hiring and consider for advanced roles if applicable.")

    elif score >= strong_score and years_exp >= strong_exp and years_exp <= max_exp_cutoff and semantic_similarity >= strong_sem_sim and (cgpa is None or cgpa >= strong_cgpa):
        overall_assessment_title = "Strong Candidate: Excellent Potential for Key Contributions"
        assessment_parts.append(f"**{candidate_name}** is a **strong candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. They show excellent alignment with the job description, supported by a solid semantic similarity of {semantic_similarity:.2f}.")
        if cgpa is not None:
            assessment_parts.append(f"Their academic performance, with a CGPA of {cgpa:.2f}, indicates a solid theoretical grounding.")
        assessment_parts.append(f"**Key Strengths:** Significant overlap in required skills and practical experience that directly addresses the job's demands. Matched keywords include: *{matched_kws_str if matched_kws_str else 'No specific keywords listed, but overall strong match'}*. This individual is likely to integrate well and contribute effectively from an early stage, bringing valuable expertise to the team.")
        assessment_parts.append("Their resume indicates a consistent track record of achieving results and adapting to new challenges. They demonstrate a solid understanding of the domain and could quickly become a valuable asset, requiring moderate onboarding.")
        assessment_parts.append("This candidate is well-suited for the role and demonstrates the core competencies required. Their experience suggests they can handle typical challenges and contribute positively to team dynamics.")
        next_steps_focus = "During the interview, explore specific project methodologies, problem-solving approaches, and long-term career aspirations to confirm alignment with team dynamics and growth opportunities within the company. Focus on behavioral questions to understand their collaboration style, initiative, and how they handle feedback. A technical assessment might be beneficial to confirm depth of skills."
        assessment_parts.append(f"**Action:** Recommend for interview. Good fit for the role, with potential for growth.")

    elif score >= promising_score and years_exp >= promising_exp and years_exp <= max_exp_cutoff and semantic_similarity >= promising_sem_sim and (cgpa is None or cgpa >= promising_cgpa):
        overall_assessment_title = "Promising Candidate: Requires Focused Review on Specific Gaps"
        assessment_parts.append(f"**{candidate_name}** is a **promising candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. While demonstrating a foundational understanding (semantic similarity: {semantic_similarity:.2f}), there are areas that warrant deeper investigation to ensure a complete fit.")
        
        gaps_identified = []
        if score < 70:
            gaps_identified.append("The overall score suggests some core skill areas may need development or further clarification.")
        if years_exp < promising_exp:
            gaps_identified.append(f"Experience ({years_exp:.1f} yrs) is on the lower side; assess their ability to scale up quickly and take on more responsibility.")
        if semantic_similarity < 0.5:
            gaps_identified.append("Semantic understanding of the JD's nuances might be limited; probe their theoretical knowledge versus practical application in real-world scenarios.")
        if cgpa is not None and cgpa < promising_cgpa:
            gaps_identified.append(f"Academic record (CGPA: {cgpa:.2f}) is below preferred, consider its relevance to role demands.")
        if missing_skills_str:
            gaps_identified.append(f"**Potential Missing Skills:** *{missing_skills_str}*. Focus interview questions on these areas to assess their current proficiency or learning agility.")
        
        if years_exp > max_exp_cutoff:
            gaps_identified.append(f"Experience ({years_exp:.1f} yrs) exceeds the maximum desired ({max_exp_cutoff} yrs). Evaluate if this indicates overqualification or a potential mismatch in role expectations.")

        if gaps_identified:
            assessment_parts.append("Areas for further exploration include: " + " ".join(gaps_identified))
        
        assessment_parts.append("The candidate shows potential, especially if they can demonstrate quick learning or relevant transferable skills. Their resume indicates a willingness to grow and take on new challenges, which is a positive sign for development opportunities.")
        next_steps_focus = "The interview should focus on validating foundational skills, understanding their learning agility, and assessing their potential for growth within the role. Be prepared to discuss specific examples of how they've applied relevant skills and how they handle challenges, particularly in areas where skills are missing. Consider a skills assessment or a structured case study to gauge problem-solving abilities. Discuss their motivation for this role and long-term career goals."
        assessment_parts.append(f"**Action:** Consider for initial phone screen or junior role. Requires careful evaluation and potentially a development plan.")

    else:
        overall_assessment_title = "Limited Match: Consider Only for Niche Needs or Pipeline Building"
        assessment_parts.append(f"**{candidate_name}** shows a **limited match** with a score = {score:.2f}% and {years_exp:.1f} years of experience (semantic similarity: {semantic_similarity:.2f}). This profile indicates a significant deviation from the core requirements of the job description.")
        if cgpa is not None:
            assessment_parts.append(f"Their academic record (CGPA: {cgpa:.2f}) also indicates a potential mismatch.")
        assessment_parts.append(f"**Key Concerns:** A low overlap in essential skills and potentially insufficient experience for the role's demands. Many key skills appear to be missing: *{missing_skills_str if missing_skills_str else 'No specific missing skills listed, but overall low match'}*. While some transferable skills may exist, a substantial investment in training or a re-evaluation of role fit would likely be required for this candidate to succeed.")
        
        if years_exp > max_exp_cutoff:
            assessment_parts.append(f"Additionally, their experience ({years_exp:.1f} yrs) significantly exceeds the maximum desired ({max_exp_cutoff} yrs), which might indicate overqualification or a mismatch in career trajectory for this specific opening.")

        assessment_parts.append("The resume does not strongly align with the technical or experience demands of this specific position. Their background may be more suited for a different type of role or industry, or an entry-level position if their core skills are strong but experience is lacking.")
        assessment_parts.append("This candidate might not be able to meet immediate role requirements without extensive support. Their current profile suggests a mismatch with the current opening.")
        next_steps_focus = "This candidate is generally not recommended for the current role unless there are specific, unforeseen niche requirements or a strategic need to broaden the candidate pool significantly. If proceeding, focus on understanding their fundamental capabilities, their motivation for this specific role despite the mismatch, and long-term career aspirations. It might be more beneficial to suggest other roles within the organization or provide feedback for future applications."
        assessment_parts.append(f"**Action:** Not recommended for this role. Consider for other open positions or future pipeline, or politely decline.")

    final_assessment = f"**Overall HR Assessment: {overall_assessment_title}**\n\n"
    final_assessment += "\n".join(assessment_parts)

    return final_assessment

def semantic_score_calculation(jd_embedding, resume_embedding, years_exp, cgpa, weighted_keyword_overlap_score, _ml_model):
    score = 0.0
    semantic_similarity = cosine_similarity(jd_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0][0]
    semantic_similarity = float(np.clip(semantic_similarity, 0, 1))

    if _ml_model is None:
        print("DEBUG: ML model not loaded in semantic_score_calculation. Providing basic score and generic feedback.")
        basic_score = (weighted_keyword_overlap_score * 0.7)
        basic_score += min(years_exp * 5, 30)
        
        if cgpa is not None:
            if cgpa >= 3.5:
                basic_score += 5
            elif cgpa < 2.5:
                basic_score -= 5
        
        score = round(min(basic_score, 100), 2)
        
        return score, round(semantic_similarity, 2)

    try:
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0
        # Ensure the feature vector matches the model's expected input shape
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        features = np.concatenate([jd_embedding, resume_embedding, [years_exp_for_model], [weighted_keyword_overlap_score]])
        
        # Reshape features to be 2D array (1 sample, N features) for .predict()
        predicted_score = _ml_model.predict(features.reshape(1, -1))[0]

        blended_score = (predicted_score * 0.6) + \
                        (weighted_keyword_overlap_score * 0.1) + \
                        (semantic_similarity * 100 * 0.3)

        if semantic_similarity > 0.7 and years_exp >= 3:
            blended_score += 5
        
        if cgpa is not None:
            if cgpa >= 3.5:
                blended_score += 3
            elif cgpa >= 3.0:
                blended_score += 1
            elif cgpa < 2.5:
                blended_score -= 2

        score = float(np.clip(blended_score, 0, 100))
        
        return round(score, 2), round(semantic_similarity, 2)

    except Exception as e:
        print(f"ERROR: Error during semantic score calculation: {e}")
        traceback.print_exc()
        basic_score = (weighted_keyword_overlap_score * 0.7)
        basic_score += min(years_exp * 5, 30)
        
        if cgpa is not None:
            basic_score += 5 if cgpa >= 3.5 else (-5 if cgpa < 2.5 else 0)

        score = round(min(basic_score, 100), 2)

        return score, 0.0

# This function is adapted from screener.py's _process_single_resume_for_screener_page
# It is now directly called for a single resume in candidate_app.py
def process_single_resume_for_candidate_app(file_name, file_bytes, file_type, jd_text, jd_name_for_results,
                                             high_priority_skills, medium_priority_skills, max_experience,
                                             _global_sentence_model, _global_ml_model):
    """
    Processes a single resume for the candidate app.
    Handles text extraction, embedding, and scoring.
    """
    try:
        # 1. Text Extraction
        text = extract_text_from_file(file_bytes, file_name, file_type)
        if text.startswith("[ERROR]"):
            return {
                "File Name": file_name,
                "Candidate Name": file_name.replace('.pdf', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').title(),
                "Score (%)": 0, "Years Experience": 0, "CGPA (4.0 Scale)": None,
                "Email": "Not Found", "Phone Number": "Not Found", "Location": "Not Found",
                "Languages": "Not Found", "Education Details": "Not Found",
                "Work History": "Not Found", "Project Details": "Not Found",
                "AI Suggestion": f"Error: {text.replace('[ERROR] ', '')}",
                "Detailed HR Assessment": f"Error processing resume: {text.replace('[ERROR] ', '')}",
                "Matched Keywords": "", "Missing Skills": "",
                "Matched Keywords (Categorized)": "{}", # Store as empty JSON string
                "Missing Skills (Categorized)": "{}", # Store as empty JSON string
                "Semantic Similarity": 0.0, "Resume Raw Text": "",
                "JD Used": jd_name_for_results, "Date Screened": datetime.now().date(),
                "Certificate ID": str(uuid.uuid4()), "Certificate Rank": "Not Applicable",
                "Tag": "❌ Text Extraction Error"
            }

        # 2. Feature Extraction
        exp = extract_years_of_experience(text)
        email = extract_email(text)
        phone = extract_phone_number(text)
        location = extract_location(text)
        languages = extract_languages(text) 
        
        education_details_text = extract_education_text(text)
        work_history_raw = extract_work_history(text)
        project_details_raw = extract_project_details(text, MASTER_SKILLS)
        
        education_details_formatted = education_details_text
        work_history_formatted = format_work_history(work_history_raw)
        project_details_formatted = format_project_details(project_details_raw)

        candidate_name = extract_name(text) or file_name.replace('.pdf', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').title()
        cgpa = extract_cgpa(text)

        resume_raw_skills_set, resume_categorized_skills = extract_relevant_keywords(text, MASTER_SKILLS)
        jd_raw_skills_set, jd_categorized_skills = extract_relevant_keywords(jd_text, MASTER_SKILLS)

        matched_keywords = list(resume_raw_skills_set.intersection(jd_raw_skills_set))
        missing_skills = list(jd_raw_skills_set.difference(resume_raw_skills_set)) 

        # Calculate weighted keyword overlap score
        weighted_keyword_overlap_score = 0
        total_jd_skill_weight = 0
        WEIGHT_HIGH = 3
        WEIGHT_MEDIUM = 2
        WEIGHT_BASE = 1

        for jd_skill in jd_raw_skills_set:
            current_weight = WEIGHT_BASE
            if jd_skill in [s.lower() for s in high_priority_skills]:
                current_weight = WEIGHT_HIGH
            elif jd_skill in [s.lower() for s in medium_priority_skills]:
                current_weight = WEIGHT_MEDIUM
            
            total_jd_skill_weight += current_weight
            
            if jd_skill in resume_raw_skills_set:
                weighted_keyword_overlap_score += current_weight
        
        # Normalize weighted score (optional, but good for consistency)
        if total_jd_skill_weight > 0:
            weighted_keyword_overlap_score = (weighted_keyword_overlap_score / total_jd_skill_weight) * 100
        else:
            weighted_keyword_overlap_score = 0 # No JD skills, score is 0

        # 3. Embedding Generation
        jd_clean = clean_text(jd_text)
        jd_embedding = _global_sentence_model.encode([jd_clean])[0]
        resume_embedding = _global_sentence_model.encode([clean_text(text)])[0]

        # 4. Score Calculation
        score, semantic_similarity = semantic_score_calculation(
            jd_embedding, resume_embedding, exp, cgpa, weighted_keyword_overlap_score, _global_ml_model
        )
        
        # 5. AI Suggestions & Assessment
        concise_ai_suggestion = generate_concise_ai_suggestion(
            candidate_name=candidate_name,
            score=score,
            years_exp=exp,
            semantic_similarity=semantic_similarity,
            cgpa=cgpa
        )

        # For candidate app, max_experience can be a fixed high number or derived from JD if JD specifies
        # For simplicity, let's use a reasonable default or a value that doesn't penalize over-experience.
        # Here we use 999 to effectively not penalize over-experience for the candidate's self-assessment.
        max_exp_for_assessment = 999 
        detailed_hr_assessment = generate_detailed_hr_assessment(
            candidate_name=candidate_name,
            score=score,
            years_exp=exp,
            semantic_similarity=semantic_similarity,
            cgpa=cgpa,
            jd_text=jd_text,
            resume_text=text,
            matched_keywords=matched_keywords,
            missing_skills=missing_skills,
            max_exp_cutoff=max_exp_for_assessment
        )

        # 6. Certificate & Tagging
        certificate_id = str(uuid.uuid4())
        certificate_rank = "Not Applicable"

        if score >= 90:
            certificate_rank = "🏅 Elite Match"
        elif score >= 80:
            certificate_rank = "⭐ Strong Match"
        elif score >= 75:
            certificate_rank = "✅ Good Fit"
        
        # Determine Tag
        tag = "❌ Limited Match"
        # Using simplified criteria for candidate's self-assessment tag
        if score >= 90:
            tag = "👑 Exceptional Match"
        elif score >= 80:
            tag = "🔥 Strong Candidate"
        elif score >= 60:
            tag = "✨ Promising Fit"
        elif score >= 40:
            tag = "⚠️ Needs Review"

        return {
            "File Name": file_name,
            "Candidate Name": candidate_name,
            "Score (%)": score,
            "Years Experience": exp,
            "CGPA (4.0 Scale)": cgpa,
            "Email": email or "Not Found",
            "Phone Number": phone or "Not Found",
            "Location": location or "Not Found",
            "Languages": languages,
            "Education Details": education_details_formatted,
            "Work History": work_history_formatted,
            "Project Details": project_details_formatted,
            "AI Suggestion": concise_ai_suggestion,
            "Detailed HR Assessment": detailed_hr_assessment,
            "Matched Keywords": ", ".join(matched_keywords),
            "Missing Skills": ", ".join(missing_skills),
            "Matched Keywords (Categorized)": json.dumps(dict(resume_categorized_skills)), # Convert to JSON string
            "Missing Skills (Categorized)": json.dumps(dict(jd_categorized_skills)),     # Convert to JSON string
            "Semantic Similarity": semantic_similarity,
            "Resume Raw Text": text, # Keep raw text for potential future use/debugging
            "JD Used": jd_name_for_results,
            "Date Screened": datetime.now().date().isoformat(), # Store as ISO format string
            "Certificate ID": certificate_id,
            "Certificate Rank": certificate_rank,
            "Tag": tag
        }
    except Exception as e:
        print(f"CRITICAL ERROR: Unhandled exception processing {file_name}: {e}")
        traceback.print_exc()
        return {
            "File Name": file_name,
            "Candidate Name": file_name.replace('.pdf', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').title(),
            "Score (%)": 0, "Years Experience": 0, "CGPA (4.0 Scale)": None,
            "Email": "Not Found", "Phone Number": "Not Found", "Location": "Not Found",
            "Languages": "Not Found", "Education Details": "Not Found",
            "Work History": "Not Found", "Project Details": "Not Found",
            "AI Suggestion": f"Critical Error: {e}",
            "Detailed HR Assessment": f"Critical Error processing resume: {e}",
            "Matched Keywords": "", "Missing Skills": "",
            "Matched Keywords (Categorized)": "{}",
            "Missing Skills (Categorized)": "{}",
            "Semantic Similarity": 0.0, "Resume Raw Text": "",
            "JD Used": jd_name_for_results, "Date Screened": datetime.now().date().isoformat(),
            "Certificate ID": str(uuid.uuid4()), "Certificate Rank": "Not Applicable",
            "Tag": "❌ Critical Processing Error"
        }

# --- Certificate Generation (adapted from screener.py) ---
@st.cache_data
def generate_certificate_html(candidate_data):
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ScreenerPro Certification - {{CANDIDATE_NAME}}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8f9fa; /* Light background for print compatibility */
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }
        .certificate-container {
            width: 100%;
            max-width: 800px;
            background: linear-gradient(145deg, #ffffff, #e6e6e6);
            border: 10px solid #00cec9; /* Main border color */
            border-radius: 20px;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .logo-text {
            font-family: 'Playfair Display', serif; /* A more elegant font for the logo */
            font-size: 2.8em; /* Larger logo text */
            color: #00cec9;
            font-weight: 700;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .certificate-header {
            font-family: 'Playfair Display', serif;
            font-size: 2.5em;
            color: #34495e; /* Darker text for header */
            margin-bottom: 20px;
            font-weight: 700;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
        }
        .certificate-subheader {
            font-size: 1.2em;
            color: #555;
            margin-bottom: 30px;
        }
        .certificate-body {
            margin-bottom: 40px;
        }
        .certificate-body p {
            font-size: 1.1em;
            line-height: 1.6;
            color: #333;
        }
        .candidate-name {
            font-family: 'Playfair Display', serif;
            font-size: 3.2em; /* Even larger name */
            color: #00cec9; /* Teal for the name */
            margin: 25px 0;
            font-weight: 700;
            border-bottom: 3px dashed #b0e0e6; /* Lighter dashed line */
            display: inline-block;
            padding-bottom: 8px;
            animation: pulse 1.5s infinite alternate; /* Subtle animation */
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            100% { transform: scale(1.02); }
        }
        .score-rank {
            font-size: 1.6em; /* Larger score/rank */
            color: #28a745; /* Green for success */
            font-weight: 700;
            margin-top: 20px;
            padding: 5px 15px;
            background-color: #e6ffe6; /* Light green background */
            border-radius: 10px;
            display: inline-block;
        }
        .date-id {
            font-size: 0.95em;
            color: #777;
            margin-top: 35px;
        }
        .footer-text {
            font-size: 0.85em;
            color: #999;
            margin-top: 40px;
        }
        /* Print styles */
        @media print {
            body {
                background-color: #fff;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
            .certificate-container {
                border: 5px solid #00cec9;
                box-shadow: none;
            }
            .logo-text, .certificate-header, .candidate-name, .score-rank {
                color: #00cec9 !important; /* Ensure colors are printed */
                text-shadow: none !important;
            }
            .candidate-name {
                animation: none !important; /* Disable animation for print */
            }
        }
    </style>
</head>
<body>
    <div class="certificate-container">
        <div class="logo-text">ScreenerPro</div>
        <div class="certificate-header">Certification of Excellence</div>
        <div class="certificate-subheader">This is to proudly certify that</div>
        <div class="candidate-name">{{CANDIDATE_NAME}}</div>
        <div class="certificate-body">
            <p>has successfully completed the comprehensive AI-powered screening process by ScreenerPro and achieved a distinguished ranking of</p>
            <div class="score-rank">
                {{CERTIFICATE_RANK}}
            </div>
            <p>demonstrating an outstanding Screener Score of **{{SCORE}}%**.</p>
            <p>This certification attests to their highly relevant skills, extensive experience, and strong alignment with the demanding requirements of modern professional roles. It signifies their readiness to excel in challenging environments and contribute significantly to organizational success.</p>
        </div>
        <div class="date-id">
            Awarded on: {{DATE_SCREENED}}<br>
            Certificate ID: {{CERTIFICATE_ID}}
        </div>
        <div class="footer-text">
            This certificate is digitally verified by ScreenerPro.
        </div>
    </div>
</body>
</html>
    """

    candidate_name = candidate_data.get('Candidate Name', 'Candidate Name')
    score = candidate_data.get('Score (%)', 0.0)
    certificate_rank = candidate_data.get('Certificate Rank', 'Not Applicable')
    date_screened_str = candidate_data.get('Date Screened')
    # Convert date_screened_str to datetime object if it's a string
    if isinstance(date_screened_str, str):
        try:
            date_screened = datetime.fromisoformat(date_screened_str).strftime("%B %d, %Y")
        except ValueError:
            date_screened = datetime.now().strftime("%B %d, %Y") # Fallback
    else: # Assume it's already a date object or similar
        date_screened = date_screened_str.strftime("%B %d, %Y") if date_screened_str else datetime.now().strftime("%B %d, %Y")

    certificate_id = candidate_data.get('Certificate ID', 'N/A')
    
    html_content = html_template.replace("{{CANDIDATE_NAME}}", candidate_name)
    html_content = html_content.replace("{{SCORE}}", f"{score:.1f}")
    html_content = html_content.replace("{{CERTIFICATE_RANK}}", certificate_rank)
    html_content = html_content.replace("{{DATE_SCREENED}}", date_screened)
    html_content = html_content.replace("{{CERTIFICATE_ID}}", certificate_id)

    return html_content

@st.cache_data
def generate_certificate_pdf(html_content):
    """Converts HTML content to PDF bytes."""
    try:
        pdf_bytes = HTML(string=html_content).write_pdf()
        return pdf_bytes
    except Exception as e:
        st.error(f"❌ Failed to generate PDF certificate: {e}")
        return None

# Modified send_certificate_email for candidate_app.py (simulation)
def send_certificate_email(recipient_email, candidate_name, score, certificate_id):
    if not recipient_email or recipient_email == "Not Found":
        st.error(f"❌ Cannot send certificate: Email address not found for {candidate_name}.")
        return False

    certificate_link = f"{CERTIFICATE_HOSTING_URL}/{certificate_id}.html" # Assuming ID maps to a hosted HTML
    
    email_subject = f"🎉 Congratulations! Your ScreenerPro Certificate is Here!"
    email_body_html = f"""
    <html>
        <body>
            <p>Hi {candidate_name},</p>
            <p>Congratulations on successfully completing the ScreenerPro resume screening process with an impressive score of <strong>{score:.1f}%</strong>!</p>
            <p>We're thrilled to present you with your official certification. This certificate recognizes your skills and employability, helping you stand out in your job search.</p>
            <p>You can view and share your certificate directly via this link:</p>
            <p><a href="{certificate_link}" style="display: inline-block; padding: 10px 20px; background-color: #00cec9; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">View Your Certificate</a></p>
            <p>Feel free to add this to your resume, LinkedIn profile, or share it with potential employers!</p>
            <p>If you have any questions, please contact us.</p>
            <p>🚀 Keep striving. Keep growing.</p>
            <p>– Team ScreenerPro</p>
        </body>
    </html>
    """
    
    st.success(f"✅ Certificate email (simulated) sent to {recipient_email}!")
    st.markdown(f"**Simulated Email Subject:** {email_subject}")
    st.markdown("**Simulated Email Body (HTML):**")
    st.components.v1.html(email_body_html, height=200, scrolling=True)
    st.info(f"The certificate link would be: {certificate_link}")
    return True

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
        st.markdown("#### Existing User Login")
        username = st.text_input("Email", key="login_email_candidate")
        password = st.text_input("Password", type="password", key="login_password_candidate")

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
        st.markdown("#### New User Registration")
        new_username = st.text_input("Email (for registration)", key="register_email_candidate")
        new_password = st.text_input("Password (for registration)", type="password", key="register_password_candidate")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_candidate")

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
    st.markdown("<h2 style='text-align: center; color: #00cec9;'>📄 AI Resume Matcher</h2>", unsafe_allow_html=True)
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
        st.markdown("### 📝 Job Description")
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
        st.markdown("### ⬆️ Upload Your Resume")
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
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("🚀 Analyze My Resume", key="analyze_candidate_resume_button", help="Click to get your resume match score!"):
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

                # Process the single resume using the adapted function
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

                st.markdown("</div>", unsafe_allow_html=True) # Close center div

                st.markdown("<h3 style='text-align: center; color: #00cec9;'>✨ Your Resume Analysis Results:</h3>", unsafe_allow_html=True)

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
                        if st.button("📧 Send Certificate to Email", key="send_cert_email_button"):
                            send_certificate_email(
                                recipient_email=screening_result['Email'],
                                candidate_name=screening_result['Candidate Name'],
                                score=screening_result['Score (%)'],
                                certificate_id=screening_result['Certificate ID']
                            )
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
                    st.markdown("### Share Your Success!")
                    linkedin_share_text = urllib.parse.quote(
                        f"I just scored {screening_result.get('Score (%)', 0):.2f}% on the ScreenerPro AI Resume Match! 🎉 Check out my certificate and try it yourself: {APP_BASE_URL} #ScreenerPro #ResumeMatch #JobSearch"
                    )
                    linkedin_share_url = f"https://www.linkedin.com/shareArticle?mini=true&url={urllib.parse.quote(APP_BASE_URL)}&title={urllib.parse.quote('ScreenerPro AI Match Certificate')}&summary={linkedin_share_text}"
                    st.markdown(f'<a href="{linkedin_share_url}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #0A66C2; color: white; border-radius: 8px; text-decoration: none; font-weight: 600;">Share on LinkedIn</a>', unsafe_allow_html=True)

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
.stSelectbox > div > div > div > div {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    color: {'#E0E0E0' if dark_mode else '#333333'};
    border-radius: 8px;
}}
.stSelectbox > label {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stExpander {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,{'0.2' if dark_mode else '0.05'});
}}
.stExpander > div > div > p {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
</style>
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
        st.markdown("<h2 style='text-align: center; color: #00cec9;'>⭐ Top Candidates This Week</h2>", unsafe_allow_html=True)
        st.info("See who's acing their resume matches!")
        top_candidates_data = load_top_candidates_leaderboard_local() # Changed to load_top_candidates_leaderboard_local
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
                f"I just checked my resume match on ScreenerPro and it's awesome! Check out the top candidates here: {APP_BASE_URL} #ScreenerPro #JobSearch #ResumeTips"
            )
            x_share_url = f"https://twitter.com/intent/tweet?text={x_share_text}"
            st.markdown(f'<a href="{x_share_url}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #1DA1F2; color: white; border-radius: 8px; text-decoration: none; font-weight: 600; margin-right: 10px;">Share on X/Twitter</a>', unsafe_allow_html=True)
            
            # LinkedIn share (already defined in certificate, but can be generic here)
            linkedin_share_text_generic = urllib.parse.quote(
                f"Check out the top candidates on ScreenerPro's AI Resume Match platform! Find your fit: {APP_BASE_URL} #ScreenerPro #ResumeMatch #JobSearch"
            )
            linkedin_share_url_generic = f"https://www.linkedin.com/shareArticle?mini=true&url={urllib.parse.quote(APP_BASE_URL)}&title={urllib.parse.quote('ScreenerPro Top Candidates')}&summary={linkedin_share_text_generic}"
            st.markdown(f'<a href="{linkedin_share_url_generic}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #0A66C2; color: white; border-radius: 8px; text-decoration: none; font-weight: 600;">Share on LinkedIn</a>', unsafe_allow_html=True)

        else:
            st.info("No top candidates to display yet. Be the first to get a high score!")

    elif candidate_tab == "🤝 Refer a Friend":
        st.markdown("<h2 style='text-align: center; color: #00cec9;'>🤝 Refer a Friend</h2>", unsafe_allow_html=True)
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

