from flask import Flask, request, render_template, send_from_directory
import pdfplumber
import docx
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import json
import os

app = Flask(__name__)

# Load spaCy model (run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# Skill improvement suggestions database
SKILL_SUGGESTIONS = {
    'python': {
        'title': 'Master Python Programming',
        'description': 'Python is fundamental for data science, automation, and backend development. Strengthen your Python fundamentals and build real projects.',
        'resources': ['LeetCode Python problems', 'Python official documentation', 'Real Python tutorials', 'Build 3 mini projects'],
        'impact_score': 15
    },
    'javascript': {
        'title': 'Develop JavaScript Expertise',
        'description': 'JavaScript is essential for frontend development. Focus on ES6+, async programming, and DOM manipulation.',
        'resources': ['MDN Web Docs', 'JavaScript.info', 'Build a full-stack project', 'Learn TypeScript'],
        'impact_score': 14
    },
    'react': {
        'title': 'Learn React Framework',
        'description': 'React is the most popular frontend framework. Master components, hooks, state management, and build a portfolio project.',
        'resources': ['React official docs', 'Scrimba React course', 'Build projects on CodePen', 'Learn Next.js'],
        'impact_score': 13
    },
    'sql': {
        'title': 'Advance SQL Skills',
        'description': 'Master SQL for database management. Learn complex queries, optimization, and work with real databases.',
        'resources': ['SQL Tutorial HackerRank', 'Database design courses', 'PostgreSQL documentation', 'Practice on LeetCode'],
        'impact_score': 12
    },
    'docker': {
        'title': 'Containerization with Docker',
        'description': 'Learn Docker for deployment and containerization. Essential for DevOps and modern development workflows.',
        'resources': ['Docker official documentation', 'Play with Docker labs', 'Build containerized apps', 'Learn Kubernetes basics'],
        'impact_score': 11
    },
    'aws': {
        'title': 'Cloud Computing with AWS',
        'description': 'Master AWS services for cloud deployment. Learn EC2, S3, Lambda, and RDS for scalable applications.',
        'resources': ['AWS free tier', 'AWS certification courses', 'Build on AWS', 'AWS architecture labs'],
        'impact_score': 11
    },
    'kubernetes': {
        'title': 'Learn Kubernetes Orchestration',
        'description': 'Master Kubernetes for container orchestration in production environments.',
        'resources': ['Kubernetes official docs', 'Play with Kubernetes', 'Certified Kubernetes courses', 'Deploy projects on K8s'],
        'impact_score': 10
    },
    'machine learning': {
        'title': 'Machine Learning Fundamentals',
        'description': 'Build a strong foundation in ML algorithms, libraries, and practical implementations.',
        'resources': ['Andrew Ng ML course', 'Scikit-learn documentation', 'Kaggle competitions', 'Build ML projects'],
        'impact_score': 12
    },
    'tensorflow': {
        'title': 'Deep Learning with TensorFlow',
        'description': 'Master TensorFlow for deep learning and neural networks. Build real-world AI solutions.',
        'resources': ['TensorFlow tutorials', 'Fast.ai course', 'Keras documentation', 'Kaggle datasets'],
        'impact_score': 13
    },
    'git': {
        'title': 'Version Control & Git Workflow',
        'description': 'Master Git and GitHub for professional version control, collaboration, and code management.',
        'resources': ['GitHub Learning Lab', 'Git documentation', 'Branching strategies tutorial', 'Practice on real projects'],
        'impact_score': 9
    },
    'rest api': {
        'title': 'REST API Development',
        'description': 'Learn to design and build RESTful APIs. Essential for backend and full-stack development.',
        'resources': ['REST API course', 'Postman tutorials', 'API design best practices', 'Build your own API'],
        'impact_score': 11
    },
    'agile': {
        'title': 'Agile & Scrum Methodology',
        'description': 'Understand Agile principles, Scrum framework, and effective team collaboration in software development.',
        'resources': ['Scrum.org resources', 'Agile certification courses', 'Participate in Scrum teams', 'Practice Scrum exercises'],
        'impact_score': 8
    },
    'java': {
        'title': 'Java Programming Mastery',
        'description': 'Deepen your Java knowledge for enterprise applications and backend development.',
        'resources': ['Oracle Java tutorials', 'Spring framework guide', 'Java concurrency guides', 'Build backend projects'],
        'impact_score': 12
    },
    'angular': {
        'title': 'Master Angular Framework',
        'description': 'Learn Angular for building scalable frontend applications with TypeScript.',
        'resources': ['Angular official docs', 'Angular University courses', 'Build projects', 'Learn RxJS'],
        'impact_score': 12
    },
    'typescript': {
        'title': 'TypeScript for Strong Typing',
        'description': 'Master TypeScript for safer, more maintainable JavaScript code.',
        'resources': ['TypeScript handbook', 'TypeScript courses', 'Refactor projects to TS', 'Advanced TS patterns'],
        'impact_score': 10
    },
    'nodejs': {
        'title': 'Backend with Node.js',
        'description': 'Build server-side applications using Node.js and Express. Essential for full-stack development.',
        'resources': ['Node.js documentation', 'Express guide', 'Build REST APIs', 'Learn async patterns'],
        'impact_score': 12
    },
    'mongodb': {
        'title': 'NoSQL Databases with MongoDB',
        'description': 'Master MongoDB for flexible, scalable database solutions in modern applications.',
        'resources': ['MongoDB University', 'MongoDB documentation', 'Build projects with MongoDB', 'Query optimization'],
        'impact_score': 10
    },
    'communication': {
        'title': 'Professional Communication Skills',
        'description': 'Develop strong verbal, written, and presentation skills for tech roles.',
        'resources': ['Toastmasters clubs', 'Technical writing courses', 'Present at meetups', 'Practice documentation'],
        'impact_score': 8
    },
    'project management': {
        'title': 'Project Management Skills',
        'description': 'Learn to lead projects, manage timelines, and coordinate teams effectively.',
        'resources': ['Project Management Institute', 'Coursera courses', 'Lead projects', 'Learn tools like Jira'],
        'impact_score': 9
    }
}

# Regex patterns for common technical skills
TECH_PATTERNS = [
    # Programming & Tech
    r'\bc\b', r'\bc\+\+\b', r'\bc#\b',
    r'\bpython\b', r'\bsql\b', r'\bjava\b',
    r'\bjavascript\b', r'\btypescript\b', r'\bruby\b',
    r'\bgo\b', r'\brust\b', r'\bswift\b', r'\bkotlin\b', r'\bphp\b',

    # Frameworks & Libraries
    r'\btensorflow\b', r'\bpytorch\b', r'\bkeras\b', r'\bsklearn\b',
    r'\bpandas\b', r'\bnumpy\b', r'\bflask\b', r'\bdjango\b',
    r'\breact\b', r'\bangular\b', r'\bvue\b',

    # Cloud & DevOps
    r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bdocker\b',
    r'\bkubernetes\b', r'\bjenkins\b', r'\bgit\b', r'\blinux\b',

    # Data & Analytics
    r'\btableau\b', r'\bpower bi\b', r'\bexcel\b',
    r'\bhadoop\b', r'\bspark\b', r'\bmachine learning\b',
    r'\bdata science\b', r'\bnatural language processing\b',

    # Productivity & Office Tools
    r'\bmicrosoft office\b', r'\bmicrosoft word\b',
    r'\bmicrosoft excel\b', r'\bmicrosoft powerpoint\b',
    r'\bmicrosoft access\b', r'\bmicrosoft project\b',
    r'\bmicrosoft visio\b', r'\bgoogle docs\b',

    # Engineering & CAD Software
    r'\bautocad\b', r'\bsolidworks\b', r'\bcatia\b',
    r'\bansys\b', r'\bmatlab\b', r'\bsimulink\b',
    r'\bcreo\b', r'\binventor\b', r'\brevit\b', r'\barchicad\b',

    # Business & Finance Software
    r'\bsap\b', r'\boracle financials\b', r'\bquickbooks\b',
    r'\bpeachtree\b', r'\bzoho books\b',

    # Statistics & Research Tools
    r'\bstata\b', r'\bspss\b', r'\brapidminer\b',
    r'\bsas\b', r'\br\b',

    # Design & Creative Software
    r'\badobe photoshop\b', r'\badobe illustrator\b',
    r'\badobe indesign\b', r'\badobe xd\b',
    r'\bfigma\b', r'\bsketch\b', r'\bcoreldraw\b',
    r'\bblender\b', r'\bmaya\b',

    # Project & Collaboration Tools
    r'\bjira\b', r'\bconfluence\b', r'\btrello\b',
    r'\basana\b', r'\bslack\b', r'\bnotion\b',

    # Specialized Fields
    r'\bepic systems\b', r'\bmeditech\b'   # healthcare IT
]
def extract_required_skills(job_description):
    text = job_description.lower()
    found_skills = set()

    # Regex-based detection
    for pattern in TECH_PATTERNS:
        matches = re.findall(pattern, text)
        found_skills.update(matches)

    # Optional: spaCy NER for extra tech/product mentions
    if nlp:
        doc = nlp(job_description)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT","SKILL"):
                found_skills.add(ent.text.lower())

    return list(found_skills)

def parse_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return text.lower()

def parse_docx(file):
    doc = docx.Document(file)
    text = " ".join([para.text for para in doc.paragraphs])
    return text.lower()

def calculate_combined_score(resume_text, job_description, required_skills):
    # Keyword match score
    matched = sorted(skill for skill in required_skills if skill in resume_text)
    missing = sorted(skill for skill in required_skills if skill not in resume_text)
    keyword_score = (len(matched) / len(required_skills)) * 100 if required_skills else 0

    # TF-IDF similarity score
    documents = [resume_text, job_description.lower()]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    # Combine both into one score (tune weights as needed)
    combined_score = (0.6 * keyword_score) + (0.4 * tfidf_score)

    return round(combined_score, 2), matched, missing

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/rank", methods=["POST"])
def rank_resumes():
    job_description = request.form.get("job_description", "")
    required_skills = extract_required_skills(job_description)

    results = []
    for file in request.files.getlist("resumes"):
        if file.filename.endswith(".pdf"):
            text = parse_pdf(file)
        elif file.filename.endswith(".docx"):
            text = parse_docx(file)
        else:
            continue

        score, matched, missing = calculate_combined_score(text, job_description, required_skills)
        results.append({
            "candidate": file.filename,
            "score": score,
            "matched": matched,
            "missing": missing,
            "shortlisted": score >= 70,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return render_template("result.html", results=results, required_skills=sorted(required_skills))


@app.route("/recruiter")
def recruiter():
    return render_template("recruiter.html")

@app.route("/job_seeker")
def job_seeker():
    return render_template("job_seeker.html")


@app.route("/service-worker.js")
def service_worker():
    return send_from_directory("static", "service-worker.js")

if __name__ == "__main__":
    app.run(debug=True)