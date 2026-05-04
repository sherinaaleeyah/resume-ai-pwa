from flask import Flask, request, render_template
import pdfplumber
import docx
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load spaCy model (run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

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

if __name__ == "__main__":
    app.run(debug=True)
