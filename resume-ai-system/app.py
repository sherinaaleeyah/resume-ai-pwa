from flask import Flask, abort, redirect, request, render_template, send_from_directory, url_for
import pdfplumber
import docx
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, "hiring_dashboard.db")

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

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()]
        if "active" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                candidate_name TEXT NOT NULL,
                file_name TEXT NOT NULL,
                score REAL NOT NULL,
                matched_skills TEXT NOT NULL,
                missing_skills TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs (id)
            )
            """
        )
        score_columns = [row["name"] for row in conn.execute("PRAGMA table_info(candidate_scores)").fetchall()]
        if "summary" not in score_columns:
            conn.execute("ALTER TABLE candidate_scores ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                message TEXT NOT NULL,
                score REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs (id)
            )
            """
        )

init_db()

def candidate_name_from_filename(filename):
    stem = Path(filename).stem
    cleaned = re.sub(r"[_-]+", " ", stem)
    cleaned = re.sub(r"\b(resume|cv|profile)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else filename

def match_strength(score):
    if score >= 85:
        return "Strong match"
    if score >= 70:
        return "Good match"
    if score >= 50:
        return "Moderate match"
    return "Needs review"

def candidate_summary(candidate_name, score, matched, missing):
    strength = match_strength(score).lower()
    matched_text = ", ".join(matched[:4]) if matched else "the available role language"
    if missing:
        gap_text = f" Main gaps to review: {', '.join(missing[:3])}."
    else:
        gap_text = " No required skills are missing from the parsed resume."
    return (
        f"{candidate_name} is a {strength} for this role with a {round(score)} match score. "
        f"The resume aligns with {matched_text}.{gap_text}"
    )

def format_activity_date(value):
    try:
        return datetime.fromisoformat(value).strftime("%d %b")
    except ValueError:
        return value

def build_recruiter_overview():
    with get_db_connection() as conn:
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
            WHERE j.active = 1
            GROUP BY j.id
            ORDER BY j.id DESC
            """
        ).fetchall()
        total_candidates = conn.execute("SELECT COUNT(*) AS total FROM candidate_scores").fetchone()["total"]
        average_match = conn.execute("SELECT AVG(score) AS average FROM candidate_scores").fetchone()["average"]
        top_candidate = conn.execute(
            """
            SELECT cs.candidate_name, cs.score, j.title
            FROM candidate_scores cs
            JOIN jobs j ON j.id = cs.job_id
            ORDER BY cs.score DESC, cs.id DESC
            LIMIT 1
            """
        ).fetchone()
        activities = conn.execute(
            """
            SELECT message, score, created_at
            FROM activity
            ORDER BY id DESC
            LIMIT 8
            """
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

def get_job_metrics(job_id):
    with get_db_connection() as conn:
        metrics = conn.execute(
            """
            SELECT
                COUNT(id) AS total_candidates,
                MAX(score) AS top_score,
                AVG(score) AS average_score
            FROM candidate_scores
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

    return {
        "total_candidates": metrics["total_candidates"],
        "top_score": round(metrics["top_score"]) if metrics["top_score"] is not None else 0,
        "average_score": round(metrics["average_score"]) if metrics["average_score"] is not None else 0,
    }

def get_ranked_candidates(job_id):
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, candidate_name, file_name, score, matched_skills, missing_skills, summary
            FROM candidate_scores
            WHERE job_id = ?
            ORDER BY score DESC, id DESC
            """,
            (job_id,),
        ).fetchall()

    candidates = []
    for row in rows:
        matched = json.loads(row["matched_skills"] or "[]")
        missing = json.loads(row["missing_skills"] or "[]")
        candidates.append({
            "id": row["id"],
            "candidate_name": row["candidate_name"],
            "file_name": row["file_name"],
            "score": round(row["score"]),
            "matched": matched,
            "missing": missing,
            "summary": row["summary"] or candidate_summary(row["candidate_name"], row["score"], matched, missing),
            "strength": match_strength(row["score"]),
        })
    return candidates

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


@app.route("/recruiter", methods=["GET", "POST"])
def recruiter():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        company = request.form.get("company", "").strip()
        description = request.form.get("description", "").strip()

        if title and company and description:
            with get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO jobs (title, company, description, created_at, active)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (title, company, description, datetime.utcnow().isoformat(timespec="seconds")),
                )
                job_id = cursor.lastrowid
                conn.execute(
                    "INSERT INTO activity (job_id, message, score, created_at) VALUES (?, ?, ?, ?)",
                    (
                        job_id,
                        f"{title} opened at {company}",
                        None,
                        datetime.utcnow().isoformat(timespec="seconds"),
                    ),
                )
            return redirect(url_for("job_detail", job_id=job_id))

    kpis, jobs = build_recruiter_overview()

    return render_template("recruiter.html", jobs=jobs, kpis=kpis)

@app.route("/recruiter/jobs/<int:job_id>")
def job_detail(job_id):
    with get_db_connection() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ? AND active = 1", (job_id,)).fetchone()

    if job is None:
        abort(404)

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
def edit_job(job_id):
    title = request.form.get("title", "").strip()
    company = request.form.get("company", "").strip()
    description = request.form.get("description", "").strip()

    if not title or not company or not description:
        return redirect(url_for("job_detail", job_id=job_id))

    with get_db_connection() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ? AND active = 1", (job_id,)).fetchone()
        if job is None:
            abort(404)

        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        conn.execute(
            "UPDATE jobs SET title = ?, company = ?, description = ? WHERE id = ?",
            (title, company, description, job_id),
        )
        conn.execute(
            "INSERT INTO activity (job_id, message, score, created_at) VALUES (?, ?, ?, ?)",
            (job_id, f"{title} job description updated", None, timestamp),
        )

    return redirect(url_for("job_detail", job_id=job_id))

@app.route("/recruiter/jobs/<int:job_id>/delete", methods=["POST"])
def delete_job(job_id):
    with get_db_connection() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ? AND active = 1", (job_id,)).fetchone()
        if job is None:
            abort(404)

        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        conn.execute("DELETE FROM candidate_scores WHERE job_id = ?", (job_id,))
        conn.execute("UPDATE jobs SET active = 0 WHERE id = ?", (job_id,))
        conn.execute(
            "INSERT INTO activity (job_id, message, score, created_at) VALUES (?, ?, ?, ?)",
            (job_id, f"{job['title']} archived and candidate data cleared", None, timestamp),
        )

    return redirect(url_for("recruiter"))

@app.route("/recruiter/jobs/<int:job_id>/candidate/<int:candidate_id>/delete", methods=["POST"])
def delete_candidate(job_id, candidate_id):
    with get_db_connection() as conn:
        candidate = conn.execute(
            """
            SELECT cs.candidate_name, cs.score, j.title
            FROM candidate_scores cs
            JOIN jobs j ON j.id = cs.job_id
            WHERE cs.job_id = ? AND cs.id = ? AND j.active = 1
            """,
            (job_id, candidate_id),
        ).fetchone()

        if candidate is None:
            abort(404)

        conn.execute("DELETE FROM candidate_scores WHERE job_id = ? AND id = ?", (job_id, candidate_id))

        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO activity (job_id, message, score, created_at) VALUES (?, ?, ?, ?)",
            (
                job_id,
                f"{candidate['candidate_name']} removed from {candidate['title']}",
                candidate["score"],
                timestamp,
            ),
        )

    return redirect(url_for("job_detail", job_id=job_id))

@app.route("/recruiter/jobs/<int:job_id>/rank", methods=["POST"])
def rank_job_resumes(job_id):
    with get_db_connection() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if job is None:
        abort(404)

    job_description = job["description"]
    required_skills = extract_required_skills(job_description)
    submitted_names = [name.strip() for name in request.form.getlist("candidate_names")]

    results = []
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    for index, file in enumerate(request.files.getlist("resumes")):
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            text = parse_pdf(file)
        elif filename.endswith(".docx"):
            text = parse_docx(file)
        else:
            continue

        score, matched, missing = calculate_combined_score(text, job_description, required_skills)
        candidate_name = submitted_names[index] if index < len(submitted_names) and submitted_names[index] else candidate_name_from_filename(file.filename)
        summary = candidate_summary(candidate_name, score, matched, missing)
        results.append({
            "candidate": file.filename,
            "candidate_name": candidate_name,
            "score": score,
            "matched": matched,
            "missing": missing,
            "summary": summary,
            "shortlisted": score >= 70,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        with get_db_connection() as conn:
            for result in results:
                conn.execute(
                    """
                    INSERT INTO candidate_scores
                        (job_id, candidate_name, file_name, score, matched_skills, missing_skills, summary, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        result["candidate_name"],
                        result["candidate"],
                        result["score"],
                        json.dumps(result["matched"]),
                        json.dumps(result["missing"]),
                        result["summary"],
                        timestamp,
                    ),
                )
                conn.execute(
                    "INSERT INTO activity (job_id, message, score, created_at) VALUES (?, ?, ?, ?)",
                    (
                        job_id,
                        f"{result['candidate_name']} added to {job['title']}",
                        result["score"],
                        timestamp,
                    ),
                )

    return render_template(
        "result.html",
        results=results,
        required_skills=sorted(required_skills),
        job=job,
    )

@app.route("/job_seeker")
def job_seeker():
    return render_template("job_seeker.html")


@app.route("/service-worker.js")
def service_worker():
    return send_from_directory("static", "service-worker.js")

if __name__ == "__main__":
    app.run(debug=True)
