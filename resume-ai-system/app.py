from flask import Flask, abort, redirect, request, render_template, send_from_directory, url_for, send_file
from flask_login import current_user, login_required
from io import BytesIO
import pdfplumber
import docx
import re
try:
    import spacy
except Exception:
    spacy = None
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _bert_model = None
import json
import os
from datetime import datetime
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import time
from google.api_core import exceptions
from werkzeug.utils import secure_filename

import auth
from auth import csrf
import db
from db import DEFAULT_KEYWORD_WEIGHT, get_db

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash")

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Recruiter login, CSRF protection and secure cookies (see auth.py).
auth.init_app(app)

# Largest single request (all files in one upload together). Each resume is also capped
# individually at MAX_RESUME_BYTES below.
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
MAX_RESUME_BYTES = 5 * 1024 * 1024
RESUME_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# Load spaCy model only if available
if spacy:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
else:
    nlp = None

# Comprehensive technical skill regex patterns
TECH_PATTERNS = [

    # =========================
    # Programming Languages
    # =========================
    # C, C++ and C# are separate languages: "c" must not match inside "c++"/"c#" (or "c&i", "c/o").
    # \b can't anchor after "+" or "#", so these use explicit lookarounds.
    r'(?<![\w+#&./])c(?![\w+#&/])', r'(?<![\w+#])c\+\+(?![\w+#])', r'(?<![\w+#])c#(?![\w+#])',
    r'\bpython\b', r'\bjava\b',
    r'\bjavascript\b', r'\btypescript\b',
    r'\bruby\b', r'\bgo\b', r'\brust\b',
    r'\bswift\b', r'\bkotlin\b',
    r'\bphp\b', r'\bperl\b',
    r'(?<![\w+#&./])r(?![\w+#&/])', r'\bmatlab\b',  # R the language, not "R&D"
    r'\bscala\b', r'\bdart\b',
    r'\b(?:vb|visual basic) ?\.net\b',
    r'\bsql\b', r'\bnosql\b',

    # =========================
    # Web Development
    # =========================
    r'\bhtml\b', r'\bhtml5\b',
    r'\bcss\b', r'\bcss3\b',
    r'\bsass\b', r'\bscss\b',
    r'\bbootstrap\b', r'\btailwind\b',
    r'\bmaterial ui\b', r'\bchakra ui\b',
    r'\bado ?\.net\b', r'\bentity framework\b',

    r'\breact\b', r'\breact\.?js\b',
    r'\bnext\.?js\b',
    r'\bangular\b',
    r'\bvue\b', r'\bvue\.?js\b',
    r'\bnuxt\.?js\b',
    r'\bsvelte\b',
    r'\bjquery\b',
    r'\bredux\b',

    r'\bnode\.?js\b',
    r'\bexpress\.?js\b',
    r'\bnest\.?js\b',
    r'\bfastapi\b',
    r'\bflask\b',
    r'\bdjango\b',
    r'\blaravel\b',
    r'\bspring boot\b',

    r'\brest api\b',
    r'\bgraphql\b',
    r'\bjson\b',
    r'\bxml\b',

    # =========================
    # Mobile Development
    # =========================
    r'\bflutter\b',
    r'\breact native\b',
    r'\bandroid studio\b',
    r'\bxcode\b',
    r'\bionic\b',
    r'\bxamarin\b',

    # =========================
    # Databases
    # =========================
    r'\bmysql\b',
    r'\bpostgresql\b',
    r'\bmongodb\b',
    r'\bsqlite\b',
    r'\boracle\b',
    r'\bredis\b',
    r'\bcassandra\b',
    r'\bmariadb\b',
    r'\belasticsearch\b',
    r'\bfirebase\b',
    r'\bsupabase\b',

    # =========================
    # AI / Machine Learning
    # =========================
    r'\bmachine learning\b',
    r'\bdeep learning\b',
    r'\bdata science\b',
    r'\bnatural language processing\b',
    r'\bnlp\b',
    r'\bcomputer vision\b',
    r'\bneural network\b',
    r'\bcnn\b',
    r'\brnn\b',
    r'\blstm\b',
    r'\btransformer\b',
    r'\byolo\b',

    r'\btensorflow\b',
    r'\bpytorch\b',
    r'\bkeras\b',
    r'\bscikit[- ]learn\b',
    r'\bsklearn\b',
    r'\bpandas\b',
    r'\bnumpy\b',
    r'\bopencv\b',
    r'\bspacy\b',
    r'\bmatplotlib\b',
    r'\bjupyter(?: notebooks?)?\b',
    r'\bhugging face\b',
    r'\blangchain\b',
    r'\bopenai\b',
    r'\bchatgpt\b',
    r'\bgemini(?: api)?\b',
    r'\bllm\b',
    r'\blogistic regression\b',
    r'\bdecision trees?\b',
    r'\bgradient boosting\b',
    r'\blightgbm\b',
    r'\bpredictive model(?:s|ing|ling)?\b',
    r'\bsentiment analysis\b',
    r'\btwitter api\b',

    # =========================
    # Cloud / DevOps
    # =========================
    r'\baws\b',
    r'\bazure\b',
    r'\bgcp\b',
    r'\bdocker\b',
    r'\bkubernetes\b',
    r'\bterraform\b',
    r'\bansible\b',
    r'\bjenkins\b',
    r'\bgithub actions\b',
    r'\bgitlab ci\b',
    r'\bcircleci\b',
    r'\btravis ci\b',
    r'\bvagrant\b',
    r'\bhelm\b',

    r'\bgit\b',
    r'\bgithub\b',
    r'\bbitbucket\b',

    # =========================
    # Operating Systems
    # =========================
    r'\blinux\b',
    r'\bubuntu\b',
    r'\bunix\b',
    r'\bwindows server\b',
    r'\bmacos\b',

    # =========================
    # Cybersecurity
    # =========================
    r'\bkali linux\b',
    r'\bwireshark\b',
    r'\bnmap\b',
    r'\bmetasploit\b',
    r'\bburp suite\b',
    r'\bnessus\b',
    r'\bsplunk\b',
    r'\bsiem\b',

    r'\bethical hacking\b',
    r'\bpenetration testing\b',
    r'\bnetwork security\b',
    r'\bcybersecurity\b',
    r'\bfirewall\b',
    r'\bincident response\b',
    r'\bdigital forensics\b',

    # =========================
    # Networking
    # =========================
    r'\bnetworking\b',
    r'\btcp/ip\b',
    r'\bosi model\b',
    r'\brouter\b',
    r'\bswitch\b',
    r'\bcisco\b',
    r'\bpacket tracer\b',
    r'\bactive directory\b',

    # =========================
    # Data Engineering / Big Data
    # =========================
    r'\bhadoop\b',
    r'\bspark\b',
    r'\bpyspark\b',
    r'\bkafka\b',
    r'\bairflow\b',
    r'\betl\b',
    r'\bdata warehouse\b',
    r'\bsnowflake\b',
    r'\bdatabricks\b',
    r'\baws glue\b',
    r'\bdata clean(?:ing|sing)\b',
    r'\bdata wrangl(?:e|es|ing)\b',

    # =========================
    # Data Analytics / BI
    # =========================
    r'\btableau\b',
    r'\bpower\s?bi\b',
    r'\bexcel\b',
    r'\bgoogle sheets\b',
    r'\bdata analysis\b',
    r'\bdata visuali[sz]ations?\b',

    # =========================
    # Testing / QA
    # =========================
    r'\bselenium\b',
    r'\bcypress\b',
    r'\bjest\b',
    r'\bmocha\b',
    r'\bpytest\b',
    r'\bunit testing\b',
    r'\bautomation testing\b',
    r'\bsoftware testing\b',

    # =========================
    # UI/UX & Creative Design
    # =========================
    r'\bfigma\b',
    r'\bsketch\b',
    r'\badobe xd\b',
    r'\bwireframing\b',
    r'\bprototyping\b',

    r'\badobe photoshop\b',
    r'\badobe illustrator\b',
    r'\badobe indesign\b',
    r'\badobe premiere\b',
    r'\bafter effects\b',
    r'\bcanva\b',
    r'\bcoreldraw\b',
    r'\bblender\b',
    r'\bmaya\b',

    # =========================
    # Engineering / CAD
    # =========================
    r'\bautocad\b',
    r'\bsolidworks\b',
    r'\bcatia\b',
    r'\bansys\b',
    r'\bsimulink\b',
    r'\bcreo\b',
    r'\binventor\b',
    r'\brevit\b',
    r'\barchicad\b',
    r'\bprota ?structure\b',
    r'\bautodesk robot\b',
    r'\brobot structural analysis\b',
    r'\bthonny\b',
   

    # =========================
    # Business / Finance / ERP
    # =========================
    r'\bsap\b',
    r'\boracle financials\b',
    r'\bquickbooks\b',
    r'\bzoho books\b',
    r'\bsalesforce\b',
    r'\bhubspot\b',
    r'\bcrm\b',
    r'\berp\b',
    r'\bservicenow\b',

    # =========================
    # Statistics / Research
    # =========================
    r'\bstata\b',
    r'\bspss\b',
    r'\bsas\b',
    r'\brapidminer\b',
    r'\bstatistical analysis\b',

    # =========================
    # Healthcare IT
    # =========================
    r'\bepic systems\b',
    r'\bmeditech\b',
    r'\belectronic medical records\b',
    r'\bemr\b',

    # =========================
    # Project Management
    # =========================
    r'\bjira\b',
    r'\bconfluence\b',
    r'\btrello\b',
    r'\basana\b',
    r'\bslack\b',
    r'\bnotion\b',
    r'\bmonday\.com\b',
    r'\bclickup\b',
    r'\bmiro\b',
    r'\bzoom\b',
    r'\bmicrosoft teams\b',

    # =========================
    # Game Development
    # =========================
    r'\bunity\b',
    r'\bunreal engine\b',
    r'\bgodot\b',

    # =========================
    # IoT / Embedded Systems
    # =========================
    r'\barduino\b',
    r'\braspberry pi\b',
    r'\besp32\b',
    r'\bembedded systems\b',

    # =========================
    # General Technical Concepts
    # =========================
    r'\boop\b',
    r'\bobject oriented programming\b',
    r'\bdata structures\b',
    r'\balgorithms\b',
    r'\bsoftware engineering\b',
    r'\bvisual studio\b',
    r'\beclipse\b',
    r'\bblockchain\b',
    r'\bsdlc\b',
    r'\bagile\b',
    r'\bscrum\b',
    r'\bapi development\b',
    r'\bmicroservices\b'
]


def candidate_name_from_filename(filename):
    stem = Path(filename).stem
    cleaned = re.sub(r"[_-]+", " ", stem)
    cleaned = re.sub(r"\b(resume|cv|profile)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else filename

def read_uploaded_resume(file):
    """Validate one uploaded resume and return (filename, bytes, error).

    `error` is None when the file is usable, otherwise a short reason it was skipped.
    The file is kept in memory and stored in the database (resume_files table); nothing is
    written to disk.
    """
    raw_name = file.filename or ""
    extension = Path(raw_name).suffix.lower()
    stem = Path(secure_filename(Path(raw_name).stem)).stem or "resume"
    filename = f"{stem}{extension}"

    if extension not in RESUME_CONTENT_TYPES:
        return filename, None, "only PDF and DOCX files are supported"

    data = file.stream.read(MAX_RESUME_BYTES + 1)
    if len(data) > MAX_RESUME_BYTES:
        return filename, None, "larger than the 5 MB limit"
    if not data:
        return filename, None, "the file is empty"
    # PDF files start with "%PDF"; DOCX files are ZIP archives and start with "PK".
    expected_start = b"%PDF" if extension == ".pdf" else b"PK"
    if not data.startswith(expected_start):
        return filename, None, f"it does not look like a real {extension[1:].upper()} file"
    return filename, data, None

def match_strength(score):
    if score >= 85:
        return "Strong match"
    if score >= 70:
        return "Good match"
    if score >= 50:
        return "Moderate match"
    return "Needs review"

def generate_ai_summary(candidate_name, job_title, job_description, matched, missing):
    
    prompt = f"""
    Role: {job_title}
    Target Requirements: {job_description}
    
    Candidate: {candidate_name}
    Matched Skills/Experience: {", ".join(matched)}
    Missing/Unclear Areas: {", ".join(missing)}

    Task: Write a 3-4 sentence professional recruiter "assessment" summary. 
    
    Guidelines:
    1. Start with a clear verdict (e.g., "[Name] is a very strong fit" or "A solid technical match").
    2. Synthesize the matched skills into career themes (e.g., instead of saying "knows Python," say "strong technical foundation in backend automation").
    3. Call out specific high-value areas (like Design Systems, User Research, or Prototyping).
    4. Mention "minor uncertainties" or gaps in a nuanced way.
    5. Avoid bullet points; use a fluid, professional narrative tone exactly like a recruiter speaking to a hiring manager.
    """

    # Inside your loop
    try:
            response = model.generate_content(prompt)
            return response.text.strip()
            
    except exceptions.ResourceExhausted:
            # If quota is hit, return this string instead of crashing
            print("Quota exceeded, skipping AI summary for now.")
            return "Summary pending (Rate limit reached). Please refresh later."
            
    except Exception as e:
            # Catch other errors (like internet issues)
            print(f"An error occurred: {e}")
            return "Summary currently unavailable."

#Job seeker: helper function
def check_ats_structure(resume_text):
    text = resume_text.lower()

    standard_sections = {
        "Skills": ["skills", "technical skills"],
        "Education": ["education", "academic background"],
        "Experience": ["experience", "work experience", "employment history"],
        "Projects": ["projects", "project experience"],
        "Certifications": ["certifications", "certificates"]
    }

    feedback = []

    for section, keywords in standard_sections.items():
        found = any(keyword in text for keyword in keywords)

        if found:
            feedback.append({
                "section": section,
                "status": "Detected",
                "message": f"{section} section is detected."
            })
        else:
            feedback.append({
                "section": section,
                "status": "Missing",
                "message": f"{section} section is not clearly detected. Consider adding this section."
            })

    return feedback


def parse_structured_feedback(text):
    """
    Parse the structured AI feedback text into a dict with three keys:
    strengths, gaps, recommendations.
    Each value is a list of bullet point strings.
    Falls back gracefully if the headers are not found.
    """
    import re

    sections = {"strengths": [], "gaps": [], "recommendations": []}

    # Split on the known headers (case-insensitive)
    # Pattern: header line followed by bullet lines until the next header or end
    header_map = {
        r"WHAT['\u2019]?S WORKING": "strengths",
        r"GAPS\s*&\s*WEAKNESSES":    "gaps",
        r"RECOMMENDATIONS":             "recommendations",
    }

    current_key = None
    for line in text.splitlines():
        stripped = line.strip()

        matched_header = False
        for pattern, key in header_map.items():
            if re.match(pattern + r"[:\s]*$", stripped, re.IGNORECASE):
                current_key = key
                matched_header = True
                break

        if matched_header or not stripped:
            continue

        if current_key:
            # Strip leading dash/bullet/asterisk
            point = re.sub(r"^[-*\u2022]\s*", "", stripped).strip()
            # Skip lines that are just the header label repeated
            if point and len(point) > 4:
                sections[current_key].append(point)

    return sections

#Job seeker: AI Feedback Function
def generate_job_seeker_ai_feedback(score, matched, missing, ats_feedback, job_description):
    missing_sections = [
        item["section"] for item in ats_feedback if item["status"] == "Missing"
    ]

    prompt = f"""
    You are an AI resume advisor helping a job seeker improve their resume for a target job.

    Target Job Description:
    {job_description}

    Resume Match Score: {score}%

    Matched Skills:
    {", ".join(matched) if matched else "No strong matched skills detected."}

    Missing Skills:
    {", ".join(missing) if missing else "No major missing skills detected."}

    Missing ATS Sections:
    {", ".join(missing_sections) if missing_sections else "No major missing ATS sections detected."}

    Task:
    Write a structured resume analysis using EXACTLY the three section headers below.
    Each section must contain 2-4 bullet points starting with a dash (-).
    Do not add any text outside of these three sections.

    WHAT'S WORKING:
    - (strengths, matched skills, what the resume does well)

    GAPS & WEAKNESSES:
    - (missing skills, skill gaps, ATS structure issues)

    RECOMMENDATIONS:
    - (specific, practical steps the job seeker should take to improve)

    Requirements:
    - Each bullet point must be one clear, concise sentence.
    - Be honest but encouraging. Write for a student or fresh graduate.
    - Do not merge sections or add extra headers.
    - Do not use bold, asterisks, or markdown formatting inside the bullet points.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()

    except exceptions.ResourceExhausted:
        return """WHAT'S WORKING:
- Your resume has been submitted and partially matches the job description.

GAPS & WEAKNESSES:
- AI feedback is temporarily unavailable due to usage limits.
- Please review the matched and missing skills sections below for details.

RECOMMENDATIONS:
- Add any missing skills listed below to your resume if you have them.
- Use clear ATS-friendly section headings such as Skills, Experience, and Education.
- Align your wording closely with the language used in the job description."""

    except Exception as e:
        print(f"AI feedback error: {e}")
        return """WHAT'S WORKING:
- Your resume has been received and analysed.

GAPS & WEAKNESSES:
- AI feedback is currently unavailable.
- Please review the skill match results and ATS section feedback below.

RECOMMENDATIONS:
- Ensure your resume includes all required skills from the job description.
- Use standard section headings to improve ATS compatibility."""

def format_activity_date(value):
    try:
        return datetime.fromisoformat(value).strftime("%d %b")
    except ValueError:
        return value

def build_recruiter_overview():
    """Dashboard data for the signed-in recruiter only."""
    params = {"uid": current_user.id}
    with get_db() as conn:
        jobs = conn.execute(
            """
            SELECT
                j.id,
                j.title,
                j.company,
                j.description,
                j.created_at,
                COUNT(cs.id) AS total_candidates,
                MAX(cs.score) AS top_score,
                AVG(cs.score) AS average_score
            FROM jobs j
            LEFT JOIN candidate_scores cs ON cs.job_id = j.id
            WHERE j.active = 1 AND j.user_id = :uid
            GROUP BY j.id
            ORDER BY j.id DESC
            """,
            params,
        ).fetchall()
        total_candidates = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM candidate_scores cs
            JOIN jobs j ON j.id = cs.job_id
            WHERE j.user_id = :uid
            """,
            params,
        ).fetchone()["total"]
        average_match = conn.execute(
            """
            SELECT AVG(cs.score) AS average
            FROM candidate_scores cs
            JOIN jobs j ON j.id = cs.job_id
            WHERE j.user_id = :uid
            """,
            params,
        ).fetchone()["average"]
        top_candidate = conn.execute(
            """
            SELECT cs.candidate_name, cs.score, j.title
            FROM candidate_scores cs
            JOIN jobs j ON j.id = cs.job_id
            WHERE j.user_id = :uid
            ORDER BY cs.score DESC, cs.id DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        activities = conn.execute(
            """
            SELECT a.message, a.score, a.created_at
            FROM activity a
            JOIN jobs j ON j.id = a.job_id
            WHERE j.user_id = :uid
            ORDER BY a.id DESC
            LIMIT 8
            """,
            params,
        ).fetchall()

    job_cards = []
    for job in jobs:
        skills = sorted(extract_required_skills(job["description"]))[:4]
        job_cards.append({
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "skills": skills,
            "total_candidates": job["total_candidates"],
            "top_score": round(job["top_score"]) if job["top_score"] is not None else 0,
            "average_score": round(job["average_score"]) if job["average_score"] is not None else 0,
        })

    if top_candidate:
        top_candidate_card = {
            "name": top_candidate["candidate_name"],
            "score": round(top_candidate["score"]),
            "role": top_candidate["title"],
            "is_demo": False,
        }
    else:
        top_candidate_card = {
            "name": "Aiyana Brooks",
            "score": 92,
            "role": "Senior Full-Stack Engineer",
            "is_demo": True,
        }

    return {
        "open_postings": len(job_cards),
        "candidates_scored": total_candidates,
        "average_match": round(average_match) if average_match is not None else 0,
        "top_candidate": top_candidate_card,
        "activities": [
            {
                "message": activity["message"],
                "score": round(activity["score"]) if activity["score"] is not None else None,
                "date": format_activity_date(activity["created_at"]),
            }
            for activity in activities
        ],
    }, job_cards

# NOTE: get_job_metrics and get_ranked_candidates do not check who owns the job. Always call
# get_owned_job() first (the routes below do), which returns 404 for another recruiter's job.
def get_job_metrics(job_id):
    with get_db() as conn:
        metrics = conn.execute(
            """
            SELECT
                COUNT(id) AS total_candidates,
                MAX(score) AS top_score,
                AVG(score) AS average_score
            FROM candidate_scores
            WHERE job_id = :job_id
            """,
            {"job_id": job_id},
        ).fetchone()

    return {
        "total_candidates": metrics["total_candidates"],
        "top_score": round(metrics["top_score"]) if metrics["top_score"] is not None else 0,
        "average_score": round(metrics["average_score"]) if metrics["average_score"] is not None else 0,
    }

def get_ranked_candidates(job_id):
    with get_db() as conn:
        # 1. Fetch Job Details first so the AI has context
        job = conn.execute(
            "SELECT title, description FROM jobs WHERE id = :job_id",
            {"job_id": job_id},
        ).fetchone()

        # Fallback if job is missing
        job_title = job["title"] if job else "Unknown Position"
        job_desc = job["description"] if job else ""

        # 2. Fetch Candidate Scores (has_file says whether a resume file is stored for the row)
        rows = conn.execute(
            """
            SELECT cs.id, cs.candidate_name, cs.file_name, cs.score, cs.matched_skills,
                   cs.missing_skills, cs.summary, cs.keyword_score, cs.semantic_score,
                   EXISTS (SELECT 1 FROM resume_files rf WHERE rf.candidate_id = cs.id) AS has_file
            FROM candidate_scores cs
            WHERE cs.job_id = :job_id
            ORDER BY cs.score DESC, cs.id DESC
            """,
            {"job_id": job_id},
        ).fetchall()

    candidates = []
    for row in rows:
        matched = json.loads(row["matched_skills"] or "[]")
        missing = json.loads(row["missing_skills"] or "[]")

        summary = row["summary"]
        if not summary:
            summary = generate_ai_summary(
                candidate_name=row["candidate_name"],
                job_title=job_title,
                job_description=job_desc,
                matched=matched,
                missing=missing
            )

        candidates.append({
            "id": row["id"],
            "candidate_name": row["candidate_name"],
            "file_name": row["file_name"],
            "has_file": bool(row["has_file"]),
            "score": round(row["score"]),
            "matched": matched,
            "missing": missing,
            "summary": summary,
            "strength": match_strength(row["score"]),
            # Component scores let the recruiter page preview re-ranking live as the weight slider moves
            "keyword_score": row["keyword_score"],
            "semantic_score": row["semantic_score"],
        })
    return candidates

# Products written more than one way. The key is the name shown and matched; any spelling
# in its list counts as the same skill in both the job description and the resume.
SKILL_ALIASES = {
    "protastructure": ["prota structure"],
    "autodesk robot": ["robot structural analysis"],
    "decision trees": ["decision tree"],
    "predictive models": ["predictive model", "predictive modeling", "predictive modelling"],
    "data cleaning": ["data cleansing"],
    "vb.net": ["vb .net", "visual basic .net", "visual basic.net"],
    "ado.net": ["ado .net"],
    "data wrangling": ["data wrangle", "data wrangles"],
    "data visualisation": ["data visualization", "data visualisations", "data visualizations"],
    "jupyter notebook": ["jupyter", "jupyter notebooks"],
    "gemini api": ["gemini"],
}
CANONICAL_SKILL = {alias: name for name, aliases in SKILL_ALIASES.items() for alias in aliases}

def extract_required_skills(job_description):
    text = job_description.lower()
    found_skills = set()

    # Regex-based detection
    for pattern in TECH_PATTERNS:
        matches = re.findall(pattern, text)
        found_skills.update(CANONICAL_SKILL.get(m, m) for m in matches)

    # Optional: spaCy NER for extra tech/product mentions
    if nlp:
        doc = nlp(job_description)
        entities = set()
        for ent in doc.ents:
            if ent.label_ not in ("ORG", "PRODUCT", "SKILL"):
                continue
            entity = " ".join(ent.text.lower().split()).strip(" .,:;-•")
            if not entity or is_non_skill_entity(entity):
                continue
            # Skip "pytorch familiarity" when "pytorch" was already found, and "python" inside "python"
            if any(skill_in_text(skill, entity) or skill_in_text(entity, skill) for skill in found_skills):
                continue
            entities.add(entity)
        found_skills.update(remove_duplicates(entities))

    return list(found_skills)

# spaCy tags headings, action verbs and degree names as ORG/PRODUCT ("Responsibilities",
# "Collect", "Bachelor of Computer Science"). Entities containing these are not skills.
# Only applied to spaCy entities: the TECH_PATTERNS list is curated, so "software
# engineering" or "statistics" found there stay.
SECTION_HEADINGS = [
    "responsibilities", "requirements", "qualifications",
    "experience", "education", "skills", "summary",
    "about", "profile", "objective", "certifications",
    "collect", "develop", "monitor", "build", "create",
    "collaborate", "perform", "manage", "maintain",
]
EDUCATION_TERMS = [
    "bachelor", "master", "degree", "diploma",
    "computer science", "statistics", "engineering",
    "information technology",
]
NON_SKILL_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in SECTION_HEADINGS + EDUCATION_TERMS) + r")\w*"
)

def is_non_skill_entity(text):
    # Whole words, allowing endings: "developing", "bachelor's", "masters" all count
    return NON_SKILL_PATTERN.search(text.lower()) is not None

def remove_duplicates(skills):
    """Drop a skill that contains another one as a whole term ("pytorch familiarity" when "pytorch" exists)."""
    skills = set(skills)
    return [skill for skill in skills
            if not any(other != skill and skill_in_text(other, skill) for other in skills)]

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

def combine_scores(keyword_score, semantic_score, keyword_weight=None):
    if keyword_weight is None:
        keyword_weight = DEFAULT_KEYWORD_WEIGHT
    return round((keyword_weight * keyword_score) + ((1 - keyword_weight) * semantic_score), 2)

def calculate_combined_score(resume_text, job_description, required_skills, keyword_weight=None):
    score, matched, missing, _, _ = calculate_score_components(
        resume_text, job_description, required_skills, keyword_weight
    )
    return score, matched, missing

def skill_in_text(skill, text):
    """True if `skill` appears in `text` as a whole term, not inside another word.

    "r" and "c" only count as the languages, "java" doesn't match "javascript",
    and "go" doesn't match "good". Symbols are part of the term, so "c" doesn't
    match "c++" or "c#". Single letters also skip "r&d" / "c/o" style uses.
    """
    skill = skill.strip().lower()
    if not skill:
        return False
    skill = CANONICAL_SKILL.get(skill, skill)
    text = text.lower()
    for term in [skill] + SKILL_ALIASES.get(skill, []):
        if len(term) == 1:
            pattern = r"(?<![\w+#&./])" + re.escape(term) + r"(?![\w+#&/])"
        else:
            pattern = r"(?<![\w+#])" + re.escape(term) + r"(?![\w+#])"
        if re.search(pattern, text):
            return True
    return False

# all-MiniLM-L6-v2 truncates input at 256 word pieces (~180 words). To read the whole
# document, embed it in overlapping word windows and average them into one vector.
CHUNK_WORDS = 150
CHUNK_OVERLAP = 25

def embed_whole_document(text):
    words = text.split()
    if not words:
        return None
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(len(words) - CHUNK_OVERLAP, 1), step)]
    vectors = _bert_model.encode(chunks, normalize_embeddings=True)
    mean = vectors.mean(axis=0)
    norm = float((mean ** 2).sum() ** 0.5)
    return mean / norm if norm else mean

def calculate_score_components(resume_text, job_description, required_skills, keyword_weight=None):
    # Keyword match score (whole-term matching)
    matched = sorted(skill for skill in required_skills if skill_in_text(skill, resume_text))
    missing = sorted(skill for skill in required_skills if not skill_in_text(skill, resume_text))
    keyword_score = (len(matched) / len(required_skills)) * 100 if required_skills else 0

    # BERT semantic similarity over the whole resume and job description
    if _bert_model is not None:
        resume_vec = embed_whole_document(resume_text)
        job_vec = embed_whole_document(job_description.lower())
        if resume_vec is None or job_vec is None:
            semantic_score = 0.0
        else:
            semantic_score = max(float((resume_vec * job_vec).sum()), 0.0) * 100
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        documents = [resume_text, job_description.lower()]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)
        semantic_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    combined_score = combine_scores(keyword_score, semantic_score, keyword_weight)

    return combined_score, matched, missing, keyword_score, float(semantic_score)



def format_list(items, key):
    values = []
    for item in items:
        text = item.get(key, "").strip()
        if text:
            values.append(text)
    return values









def check_database_on_startup():
    """Local SQLite gets its tables automatically. PostgreSQL (Neon) must be set up once with
    `python setup_db.py`; if that was forgotten, say so clearly in the log instead of failing
    later with a confusing 'relation does not exist' error."""
    try:
        if db.is_sqlite():
            db.init_schema()
        elif not db.schema_ready():
            print("WARNING: the database has no TalentMatch tables yet. Run:  python setup_db.py")
    except Exception as e:
        print(f"WARNING: could not check the database ({e}). Is DATABASE_URL correct?")

check_database_on_startup()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


def get_owned_job(conn, job_id):
    """The job, but only if it belongs to the signed-in recruiter and has not been archived.
    Anyone else's job (or a missing one) gives the same 404, so job IDs can't be probed."""
    job = conn.execute(
        "SELECT * FROM jobs WHERE id = :job_id AND user_id = :uid AND active = 1",
        {"job_id": job_id, "uid": current_user.id},
    ).fetchone()
    if job is None:
        abort(404)
    return job


@app.route("/recruiter", methods=["GET", "POST"])
@login_required
def recruiter():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        company = request.form.get("company", "").strip()
        description = request.form.get("description", "").strip()

        if title and company and description:
            timestamp = datetime.utcnow().isoformat(timespec="seconds")
            with get_db() as conn:
                job_id = conn.execute(
                    """
                    INSERT INTO jobs (user_id, title, company, description, created_at, active)
                    VALUES (:uid, :title, :company, :description, :created_at, 1)
                    RETURNING id
                    """,
                    {
                        "uid": current_user.id,
                        "title": title,
                        "company": company,
                        "description": description,
                        "created_at": timestamp,
                    },
                ).scalar()
                conn.execute(
                    "INSERT INTO activity (job_id, message, score, created_at) VALUES (:job_id, :message, NULL, :created_at)",
                    {"job_id": job_id, "message": f"{title} opened at {company}", "created_at": timestamp},
                )
            return redirect(url_for("job_detail", job_id=job_id))

    kpis, jobs = build_recruiter_overview()

    return render_template("recruiter.html", jobs=jobs, kpis=kpis)

@app.route("/recruiter/jobs/<int:job_id>")
@login_required
def job_detail(job_id):
    with get_db() as conn:
        job = get_owned_job(conn, job_id)

    required_skills = sorted(extract_required_skills(job["description"]))
    return render_template(
        "recruiter.html",
        jobs=[],
        selected_job=job,
        required_skills=required_skills,
        job_metrics=get_job_metrics(job_id),
        ranked_candidates=get_ranked_candidates(job_id),
    )

@app.route("/recruiter/jobs/<int:job_id>/edit", methods=["POST"])
@login_required
def edit_job(job_id):
    title = request.form.get("title", "").strip()
    company = request.form.get("company", "").strip()
    description = request.form.get("description", "").strip()

    if not title or not company or not description:
        return redirect(url_for("job_detail", job_id=job_id))

    with get_db() as conn:
        job = get_owned_job(conn, job_id)

        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        conn.execute(
            "UPDATE jobs SET title = :title, company = :company, description = :description WHERE id = :job_id",
            {"title": title, "company": company, "description": description, "job_id": job_id},
        )

        message = f"{title} job description updated"
        if description != job["description"]:
            rescored, skipped = rescore_job_candidates(conn, job_id, title, description, job["keyword_weight"])
            message += f" and {rescored} candidate{'s' if rescored != 1 else ''} re-scored"
            if skipped:
                message += f" ({skipped} without a stored resume kept their old score)"

        conn.execute(
            "INSERT INTO activity (job_id, message, score, created_at) VALUES (:job_id, :message, NULL, :created_at)",
            {"job_id": job_id, "message": message, "created_at": timestamp},
        )

    return redirect(url_for("job_detail", job_id=job_id))

def rescore_job_candidates(conn, job_id, job_title, job_description, keyword_weight):
    required_skills = extract_required_skills(job_description)
    rows = conn.execute(
        "SELECT id, candidate_name, resume_text FROM candidate_scores WHERE job_id = :job_id",
        {"job_id": job_id},
    ).fetchall()

    rescored = skipped = 0
    for row in rows:
        if not row["resume_text"]:
            skipped += 1
            continue

        score, matched, missing, keyword_score, semantic_score = calculate_score_components(
            row["resume_text"], job_description, required_skills, keyword_weight
        )
        summary = generate_ai_summary(
            candidate_name=row["candidate_name"],
            job_title=job_title,
            job_description=job_description,
            matched=matched,
            missing=missing,
        )
        conn.execute(
            """
            UPDATE candidate_scores
            SET score = :score, keyword_score = :keyword_score, semantic_score = :semantic_score,
                matched_skills = :matched, missing_skills = :missing, summary = :summary
            WHERE id = :id
            """,
            {
                "score": score,
                "keyword_score": keyword_score,
                "semantic_score": semantic_score,
                "matched": json.dumps(matched),
                "missing": json.dumps(missing),
                "summary": summary,
                "id": row["id"],
            },
        )
        rescored += 1
    return rescored, skipped

@app.route("/recruiter/jobs/<int:job_id>/weights", methods=["POST"])
@login_required
def update_job_weights(job_id):
    try:
        keyword_percent = int(request.form.get("keyword_weight", ""))
    except ValueError:
        return redirect(url_for("job_detail", job_id=job_id))
    keyword_weight = max(0, min(100, keyword_percent)) / 100

    with get_db() as conn:
        job = get_owned_job(conn, job_id)

        conn.execute(
            "UPDATE jobs SET keyword_weight = :weight WHERE id = :job_id",
            {"weight": keyword_weight, "job_id": job_id},
        )

        # Re-rank existing candidates from their stored component scores
        rows = conn.execute(
            "SELECT id, keyword_score, semantic_score FROM candidate_scores WHERE job_id = :job_id",
            {"job_id": job_id},
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE candidate_scores SET score = :score WHERE id = :id",
                {"score": combine_scores(row["keyword_score"], row["semantic_score"], keyword_weight), "id": row["id"]},
            )

        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO activity (job_id, message, score, created_at) VALUES (:job_id, :message, NULL, :created_at)",
            {
                "job_id": job_id,
                "message": f"{job['title']} scoring set to {keyword_percent}% keyword / {100 - keyword_percent}% semantic",
                "created_at": timestamp,
            },
        )

    return redirect(url_for("job_detail", job_id=job_id))

@app.route("/recruiter/jobs/<int:job_id>/delete", methods=["POST"])
@login_required
def delete_job(job_id):
    with get_db() as conn:
        job = get_owned_job(conn, job_id)

        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        # Stored resume files first (they reference the candidate rows), then the candidates.
        conn.execute(
            "DELETE FROM resume_files WHERE candidate_id IN (SELECT id FROM candidate_scores WHERE job_id = :job_id)",
            {"job_id": job_id},
        )
        conn.execute("DELETE FROM candidate_scores WHERE job_id = :job_id", {"job_id": job_id})
        conn.execute("UPDATE jobs SET active = 0 WHERE id = :job_id", {"job_id": job_id})
        conn.execute(
            "INSERT INTO activity (job_id, message, score, created_at) VALUES (:job_id, :message, NULL, :created_at)",
            {"job_id": job_id, "message": f"{job['title']} archived and candidate data cleared", "created_at": timestamp},
        )

    return redirect(url_for("recruiter"))

@app.route("/recruiter/jobs/<int:job_id>/candidate/<int:candidate_id>/delete", methods=["POST"])
@login_required
def delete_candidate(job_id, candidate_id):
    with get_db() as conn:
        candidate = conn.execute(
            """
            SELECT cs.candidate_name, cs.score, j.title
            FROM candidate_scores cs
            JOIN jobs j ON j.id = cs.job_id
            WHERE cs.job_id = :job_id AND cs.id = :candidate_id AND j.user_id = :uid AND j.active = 1
            """,
            {"job_id": job_id, "candidate_id": candidate_id, "uid": current_user.id},
        ).fetchone()

        if candidate is None:
            abort(404)

        conn.execute("DELETE FROM resume_files WHERE candidate_id = :candidate_id", {"candidate_id": candidate_id})
        conn.execute(
            "DELETE FROM candidate_scores WHERE job_id = :job_id AND id = :candidate_id",
            {"job_id": job_id, "candidate_id": candidate_id},
        )

        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO activity (job_id, message, score, created_at) VALUES (:job_id, :message, :score, :created_at)",
            {
                "job_id": job_id,
                "message": f"{candidate['candidate_name']} removed from {candidate['title']}",
                "score": candidate["score"],
                "created_at": timestamp,
            },
        )

    return redirect(url_for("job_detail", job_id=job_id))

@app.route("/recruiter/jobs/<int:job_id>/candidate/<int:candidate_id>/resume")
@login_required
def download_resume(job_id, candidate_id):
    """Serve a stored resume file, but only to the recruiter who owns the job."""
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT rf.filename, rf.content_type, rf.data
            FROM resume_files rf
            JOIN candidate_scores cs ON cs.id = rf.candidate_id
            JOIN jobs j ON j.id = cs.job_id
            WHERE cs.id = :candidate_id AND cs.job_id = :job_id AND j.user_id = :uid AND j.active = 1
            """,
            {"candidate_id": candidate_id, "job_id": job_id, "uid": current_user.id},
        ).fetchone()

    if row is None:
        abort(404)

    response = send_file(
        BytesIO(bytes(row["data"])),  # PostgreSQL hands back a memoryview; bytes() makes it a normal copy
        mimetype=row["content_type"],
        download_name=row["filename"],
        as_attachment=False,
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

@app.route("/recruiter/jobs/<int:job_id>/rank", methods=["POST"])
@login_required
def rank_job_resumes(job_id):
    with get_db() as conn:
        job = get_owned_job(conn, job_id)

    job_title = job["title"]
    job_description = job["description"]

    required_skills = extract_required_skills(job_description)
    submitted_names = [name.strip() for name in request.form.getlist("candidate_names")]

    results = []
    skipped = []   # (file name, reason) for files that could not be used
    timestamp = datetime.utcnow().isoformat(timespec="seconds")

    uploaded_files = [f for f in request.files.getlist("resumes") if f and f.filename]
    for index, file in enumerate(uploaded_files):
        original_filename, data, problem = read_uploaded_resume(file)
        if problem:
            skipped.append((original_filename, problem))
            continue

        try:
            if original_filename.lower().endswith(".pdf"):
                text = parse_pdf(BytesIO(data))
            else:
                text = parse_docx(BytesIO(data))
        except Exception as e:
            print(f"Could not read {original_filename}: {e}")
            skipped.append((original_filename, "the file could not be read (damaged or password-protected?)"))
            continue

        score, matched, missing, keyword_score, semantic_score = calculate_score_components(
            text, job_description, required_skills, job["keyword_weight"]
        )

        # Determine candidate name
        candidate_name = submitted_names[index] if index < len(submitted_names) and submitted_names[index] else candidate_name_from_filename(original_filename)

        summary = generate_ai_summary(
            candidate_name=candidate_name,
            job_title=job_title,
            job_description=job_description,
            matched=matched,
            missing=missing
        )

        results.append({
            "candidate": candidate_name,
            "candidate_name": candidate_name,
            "file_name": original_filename,
            "file_url": None,          # filled in once the file is saved and has an id
            "score": score,
            "keyword_score": keyword_score,
            "semantic_score": semantic_score,
            "resume_text": text,
            "matched": matched,
            "missing": missing,
            "summary": summary,
            "shortlisted": score >= 70,
            "_file_data": data,
            "_content_type": RESUME_CONTENT_TYPES[Path(original_filename).suffix.lower()],
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    if results:
        with get_db() as conn:
            for result in results:
                candidate_id = conn.execute(
                    """
                    INSERT INTO candidate_scores
                        (job_id, candidate_name, file_name, score, keyword_score, semantic_score, resume_text,
                         matched_skills, missing_skills, summary, created_at)
                    VALUES (:job_id, :candidate_name, :file_name, :score, :keyword_score, :semantic_score, :resume_text,
                            :matched_skills, :missing_skills, :summary, :created_at)
                    RETURNING id
                    """,
                    {
                        "job_id": job_id,
                        "candidate_name": result["candidate_name"],
                        "file_name": result["file_name"],
                        "score": result["score"],
                        "keyword_score": result["keyword_score"],
                        "semantic_score": result["semantic_score"],
                        "resume_text": result["resume_text"],
                        "matched_skills": json.dumps(result["matched"]),
                        "missing_skills": json.dumps(result["missing"]),
                        "summary": result["summary"],
                        "created_at": timestamp,
                    },
                ).scalar()
                conn.execute(
                    """
                    INSERT INTO resume_files (candidate_id, filename, content_type, size_bytes, data)
                    VALUES (:candidate_id, :filename, :content_type, :size_bytes, :data)
                    """,
                    {
                        "candidate_id": candidate_id,
                        "filename": result["file_name"],
                        "content_type": result["_content_type"],
                        "size_bytes": len(result["_file_data"]),
                        "data": result["_file_data"],
                    },
                )
                conn.execute(
                    "INSERT INTO activity (job_id, message, score, created_at) VALUES (:job_id, :message, :score, :created_at)",
                    {
                        "job_id": job_id,
                        "message": f"{result['candidate_name']} added to {job_title}",
                        "score": result["score"],
                        "created_at": timestamp,
                    },
                )
                result["file_url"] = url_for("download_resume", job_id=job_id, candidate_id=candidate_id)

    for result in results:
        result.pop("_file_data", None)
        result.pop("_content_type", None)

    return render_template(
        "result.html",
        results=results,
        required_skills=sorted(required_skills),
        job=job,
        skipped=skipped,
    )

@app.route("/job_seeker")
def job_seeker():
    return render_template("job_seeker.html")

@app.route("/resume_builder")
def resume_builder():
    return render_template("resume_builder.html")

@app.route("/analyze_job_seeker_resume", methods=["POST"])
@csrf.exempt  # public job seeker tool, no login or session involved
def analyze_job_seeker_resume():
    job_description = request.form.get("job_description", "").strip()
    resume_file = request.files.get("resume")

    if not resume_file or not job_description:
        return render_template(
            "job_seeker_result.html",
            error="Please upload a resume and paste a target job description."
        )

    filename = resume_file.filename.lower()

    if filename.endswith(".pdf"):
        resume_text = parse_pdf(resume_file)
    elif filename.endswith(".docx"):
        resume_text = parse_docx(resume_file)
    else:
        return render_template(
            "job_seeker_result.html",
            error="Unsupported file format. Please upload PDF or DOCX."
        )

    required_skills = extract_required_skills(job_description)

    score, matched, missing = calculate_combined_score(
        resume_text,
        job_description,
        required_skills
    )

    ats_feedback = check_ats_structure(resume_text)

    ai_feedback_raw = generate_job_seeker_ai_feedback(
        score=score,
        matched=matched,
        missing=missing,
        ats_feedback=ats_feedback,
        job_description=job_description
    )
    ai_feedback = parse_structured_feedback(ai_feedback_raw)

    return render_template(
        "job_seeker_result.html",
        score=score,
        matched=matched,
        missing=missing,
        required_skills=sorted(required_skills),
        ats_feedback=ats_feedback,
        ai_feedback=ai_feedback
    )

def check_builder_ats_structure(resume_data):
    feedback = []

    skills = resume_data.get("skills", [])
    educations = resume_data.get("educations", [])
    experiences = resume_data.get("experiences", [])
    projects = resume_data.get("projects", [])
    certifications = resume_data.get("certifications", [])

    if skills:
        feedback.append({
            "section": "Skills",
            "status": "Detected",
            "message": "Skills section is detected."
        })
    else:
        feedback.append({
            "section": "Skills",
            "status": "Missing",
            "message": "Skills section is not clearly detected. Consider adding relevant technical and soft skills."
        })

    if educations:
        feedback.append({
            "section": "Education",
            "status": "Detected",
            "message": "Education section is detected."
        })
    else:
        feedback.append({
            "section": "Education",
            "status": "Missing",
            "message": "Education section is not clearly detected. Consider adding your school, degree, and study period."
        })

    if experiences:
        feedback.append({
            "section": "Experience",
            "status": "Detected",
            "message": "Experience section is detected."
        })
    else:
        feedback.append({
            "section": "Experience",
            "status": "Missing",
            "message": "Experience section is not clearly detected. Consider adding work experience, internship, or relevant responsibilities."
        })

    if projects:
        feedback.append({
            "section": "Projects",
            "status": "Detected",
            "message": "Projects section is detected."
        })
    else:
        feedback.append({
            "section": "Projects",
            "status": "Missing",
            "message": "Projects section is not clearly detected. Consider adding academic or personal projects."
        })

    if certifications:
        feedback.append({
            "section": "Certifications",
            "status": "Detected",
            "message": "Certifications section is detected."
        })
    else:
        feedback.append({
            "section": "Certifications",
            "status": "Missing",
            "message": "Certifications section is not clearly detected. Consider adding relevant certificates if available."
        })

    return feedback

@app.route("/download_resume_docx", methods=["POST"])
@csrf.exempt  # public job seeker tool, no login or session involved
def download_resume_docx():
    full_name = request.form.get("full_name", "Resume")
    email = request.form.get("email", "")
    phone = request.form.get("phone", "")
    location = request.form.get("location", "")
    target_role = request.form.get("target_role", "")

    summary = request.form.get("summary", "")
    skills = request.form.get("skills", "")
    experience = request.form.get("experience", "")
    education = request.form.get("education", "")
    projects = request.form.get("projects", "")
    certifications = request.form.get("certifications", "")

    document = docx.Document()

    document.add_heading(full_name, level=0)

    contact_line = " | ".join(
        item for item in [email, phone, location] if item
    )
    if contact_line:
        document.add_paragraph(contact_line)

    if target_role:
        document.add_paragraph(f"Target Role: {target_role}")

    if summary:
        document.add_heading("Professional Summary", level=1)
        document.add_paragraph(summary)

    if skills:
        document.add_heading("Skills", level=1)
        document.add_paragraph(skills)

    if experience:
        document.add_heading("Work Experience", level=1)
        document.add_paragraph(experience)

    if education:
        document.add_heading("Education", level=1)
        document.add_paragraph(education)

    if projects:
        document.add_heading("Projects", level=1)
        document.add_paragraph(projects)

    if certifications:
        document.add_heading("Certifications", level=1)
        document.add_paragraph(certifications)

    file_stream = BytesIO()
    document.save(file_stream)
    file_stream.seek(0)

    filename = f"{full_name.replace(' ', '_')}_Resume.docx"

    return send_file(
        file_stream,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

def safe_json_loads(value):
    try:
        return json.loads(value) if value else []
    except json.JSONDecodeError:
        return []

def build_resume_text_from_builder(resume_data):
    resume_parts = []

    resume_parts.append(resume_data.get("summary", ""))
    resume_parts.append(resume_data.get("ai_summary", ""))

    for skill in resume_data.get("skills", []):
        resume_parts.append(skill)

    for exp in resume_data.get("experiences", []):
        resume_parts.append(exp.get("company", ""))
        resume_parts.append(exp.get("role", ""))
        resume_parts.append(exp.get("period", ""))
        resume_parts.append(exp.get("description", ""))
        resume_parts.append(exp.get("ai_bullets", ""))

    for edu in resume_data.get("educations", []):
        resume_parts.append(edu.get("school", ""))
        resume_parts.append(edu.get("degree", ""))
        resume_parts.append(edu.get("period", ""))

    for project in resume_data.get("projects", []):
        resume_parts.append(project.get("title", ""))
        resume_parts.append(project.get("tools", ""))
        resume_parts.append(project.get("description", ""))

    for cert in resume_data.get("certifications", []):
        resume_parts.append(cert.get("name", ""))
        resume_parts.append(cert.get("provider", ""))
        resume_parts.append(cert.get("year", ""))

    return " ".join(resume_parts).lower()

def generate_ai_resume_summary(resume_data):
    prompt = f"""
    You are an AI resume writing assistant.

    Candidate Information:
    Name: {resume_data.get("full_name")}
    Target Role: {resume_data.get("target_role")}
    Current Summary: {resume_data.get("summary")}
    Skills: {", ".join(resume_data.get("skills", []))}
    Experience: {resume_data.get("experience_text")}
    Education: {resume_data.get("education_text")}
    Projects: {resume_data.get("project_text")}
    Certifications: {resume_data.get("certification_text")}

    Task:
    Rewrite the professional summary into an ATS-friendly resume summary.

    Requirements:
    1. Keep it around 2-3 sentences.
    2. Use professional language.
    3. Highlight relevant skills, education, projects, or experience.
    4. Do not invent fake experience, fake skills, or fake certifications.
    5. Make it suitable for the target role.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()

    except exceptions.ResourceExhausted:
        return resume_data.get("summary", "")

    except Exception as e:
        print(f"AI resume summary error: {e}")
        return resume_data.get("summary", "")
    
def clean_bullet_points(text):
    """
    Strip any intro sentence Gemini adds (e.g. "Here are 3 bullet points:")
    and normalise bullet markers to the standard • character.
    """
    import re
    lines = text.strip().splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip intro/outro lines that don't start with a bullet marker
        if re.match(r"^(here are|the following|below are|these are)", stripped, re.IGNORECASE):
            continue
        # Normalise *, -, – markers to •
        stripped = re.sub(r"^[\*\-\–]\s+", "• ", stripped)
        # If it already starts with • keep it, otherwise prefix it
        if not stripped.startswith("•"):
            stripped = "• " + stripped
        cleaned.append(stripped)
    return "\n".join(cleaned)

def generate_ai_bullet_points(experience):
    prompt = f"""
    You are an AI resume writing assistant.

    Convert the following work experience into exactly 3 resume bullet points.

    Company: {experience.get("company")}
    Role: {experience.get("role")}
    Period: {experience.get("period")}
    What the candidate did:
    {experience.get("description")}

    Requirements:
    1. Output ONLY the 3 bullet points, nothing else — no intro sentence, no heading, no extra text.
    2. Start each bullet point with a • symbol.
    3. Use strong action verbs.
    4. Make each bullet concise and ATS-friendly.
    5. Do not invent fake achievements or numbers.
    """

    try:
        response = model.generate_content(prompt)
        return clean_bullet_points(response.text.strip())

    except exceptions.ResourceExhausted:
        return experience.get("description", "")

    except Exception as e:
        print(f"AI bullet generation error: {e}")
        return experience.get("description", "")

def generate_builder_ai_feedback(score, matched, missing, ats_feedback, target_role):
    missing_sections = [
        item["section"] for item in ats_feedback if item["status"] == "Missing"
    ]

    prompt = f"""
    You are an AI resume advisor.

    Target Role: {target_role}
    Resume Score: {score}%

    Matched Skills:
    {", ".join(matched) if matched else "No strong matched skills detected."}

    Missing Skills:
    {", ".join(missing) if missing else "No major missing skills detected."}

    Missing ATS Sections:
    {", ".join(missing_sections) if missing_sections else "No major missing sections detected."}

    Task:
    Write a structured resume analysis using EXACTLY the three section headers below.
    Each section must contain 2-4 bullet points starting with a dash (-).
    Do not add any text outside of these three sections.

    WHAT'S WORKING:
    - (strengths, matched skills, what the resume does well)

    GAPS & WEAKNESSES:
    - (missing skills, skill gaps, ATS structure issues)

    RECOMMENDATIONS:
    - (specific, practical steps the job seeker should take to improve)

    Requirements:
    - Each bullet point must be one clear, concise sentence.
    - Be honest but encouraging. Write for a student or fresh graduate.
    - Do not merge sections or add extra headers.
    - Do not use bold, asterisks, or markdown formatting inside the bullet points.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()

    except exceptions.ResourceExhausted:
        return """WHAT'S WORKING:
- Your resume has been submitted and partially matches the target role.

GAPS & WEAKNESSES:
- AI feedback is temporarily unavailable due to usage limits.
- Please review the matched and missing skills sections below for details.

RECOMMENDATIONS:
- Add any missing skills listed below to your resume if you have them.
- Use clear ATS-friendly section headings such as Skills, Experience, and Education.
- Tailor your professional summary to the target role."""

    except Exception as e:
        print(f"AI builder feedback error: {e}")
        return """WHAT'S WORKING:
- Your resume has been received and analysed.

GAPS & WEAKNESSES:
- AI feedback is currently unavailable.
- Please review the skill match results and ATS section feedback below.

RECOMMENDATIONS:
- Ensure your resume includes all required skills from the job description.
- Use standard section headings to improve ATS compatibility."""

@app.route("/generate_resume", methods=["POST"])
@csrf.exempt  # public job seeker tool, no login or session involved
def generate_resume():
    skills = safe_json_loads(request.form.get("skills_json"))
    experiences = safe_json_loads(request.form.get("experiences_json"))
    educations = safe_json_loads(request.form.get("educations_json"))
    projects = safe_json_loads(request.form.get("projects_json"))
    certifications = safe_json_loads(request.form.get("certifications_json"))

    resume_data = {
        "full_name": request.form.get("full_name", "").strip(),
        "email": request.form.get("email", "").strip(),
        "phone": request.form.get("phone", "").strip(),
        "location": request.form.get("location", "").strip(),
        "target_role": request.form.get("target_role", "").strip(),
        "target_job_description": request.form.get("target_job_description", "").strip(),
        "summary": request.form.get("summary", "").strip(),
        "skills": skills,
        "experiences": experiences,
        "educations": educations,
        "projects": projects,
        "certifications": certifications
    }

    enhanced_experiences = []

    for exp in experiences:
        ai_bullets = generate_ai_bullet_points(exp)
        exp["ai_bullets"] = ai_bullets
        enhanced_experiences.append(exp)

    resume_data["experiences"] = enhanced_experiences

    resume_data["experience_text"] = "\n\n".join(
        [
            f"{exp.get('role', '')} at {exp.get('company', '')} ({exp.get('period', '')})\n{exp.get('ai_bullets', '')}"
            for exp in enhanced_experiences
        ]
    )

    resume_data["education_text"] = "\n".join(
        [
            f"{edu.get('degree', '')}, {edu.get('school', '')} ({edu.get('period', '')})"
            for edu in educations
        ]
    )

    resume_data["project_text"] = "\n\n".join(
        [
            f"{project.get('title', '')}\nTools: {project.get('tools', '')}\n{project.get('description', '')}"
            for project in projects
        ]
    )

    resume_data["certification_text"] = "\n".join(
        [
            f"{cert.get('name', '')}, {cert.get('provider', '')} ({cert.get('year', '')})"
            for cert in certifications
        ]
    )

    ai_summary = generate_ai_resume_summary(resume_data)
    resume_data["ai_summary"] = ai_summary

    resume_text = build_resume_text_from_builder(resume_data)

    target_job_description = resume_data.get("target_job_description", "")

    if target_job_description:
        required_skills = extract_required_skills(target_job_description)

        score, matched, missing = calculate_combined_score(
            resume_text,
            target_job_description,
            required_skills
        )
    else:
        required_skills = []
        score = 0
        matched = []
        missing = []

    ats_feedback = check_builder_ats_structure(resume_data)

    ai_feedback_raw = generate_builder_ai_feedback(
        score=score,
        matched=matched,
        missing=missing,
        ats_feedback=ats_feedback,
        target_role=resume_data.get("target_role", "")
    )
    ai_feedback = parse_structured_feedback(ai_feedback_raw)

    return render_template(
        "resume_preview.html",
        resume=resume_data,
        score=score,
        matched=matched,
        missing=missing,
        required_skills=sorted(required_skills),
        ats_feedback=ats_feedback,
        ai_feedback=ai_feedback
    )

@app.route("/service-worker.js")
def service_worker():
    return send_from_directory("static", "service-worker.js")

if __name__ == "__main__":
    # Local development only. On Azure the app is started by gunicorn.
    app.run(debug=True)