"""
Create the shared DEMO recruiter account and (optionally) import the old jobs and candidates
from the previous SQLite version of TalentMatch into it.

Usage:
    python seed_demo.py                                  # just create/update the demo account
    python seed_demo.py --source hiring_dashboard.db     # ...and import the old jobs + candidates
    python seed_demo.py --source hiring_dashboard.db --uploads uploads   # also attach the old resume files
    python seed_demo.py --source hiring_dashboard.db --uploads uploads --reset   # wipe the demo account's jobs first

The login details come from environment variables (put them in .env locally; never in the code,
because the repository is public):
    DEMO_EMAIL      e.g. demo@yourdomain.com
    DEMO_PASSWORD   a strong password, at least 8 characters
    DEMO_NAME       optional, default "Demo Recruiter"
    DEMO_COMPANY    optional, default "TalentMatch Demo"

It writes to whatever DATABASE_URL points to, so run it once against the dev database to try it,
then once against the production database.

Safe to re-run: without --reset it refuses to import a second copy of the jobs.
"""
import argparse
import glob
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from werkzeug.security import generate_password_hash

load_dotenv()

import db  # noqa: E402
from db import get_db  # noqa: E402

MAX_RESUME_BYTES = 5 * 1024 * 1024
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def find_uploaded_resume(uploads_dir, original_filename, created_at):
    """Old versions saved uploads as '<stem>_<utc timestamp><suffix>' in uploads/ and never stored
    the saved name, so match a candidate row to its file by name and time (earliest file saved
    within 10 minutes after the row was created)."""
    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix.lower()
    try:
        row_time = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return None

    best = None
    for path in Path(uploads_dir).glob(f"{glob.escape(stem)}_*{suffix}"):
        try:
            saved_time = datetime.strptime(path.stem[len(stem) + 1:], "%Y%m%d%H%M%S%f")
        except ValueError:
            continue
        delta = (saved_time.replace(microsecond=0) - row_time).total_seconds()
        if 0 <= delta <= 600 and (best is None or saved_time < best[0]):
            best = (saved_time, path)
    return best[1] if best else None


def extract_text(data, extension):
    """Same text extraction the app uses (lower-cased), so imported candidates can be re-scored."""
    import io
    if extension == ".pdf":
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return text.lower()
    import docx
    return " ".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs).lower()


def legacy_component_scores(score, matched_json, missing_json):
    """Rows saved before keyword/semantic scores were stored: rebuild them from the saved skills,
    assuming the default 60/40 weighting (this is what the old app did on startup)."""
    import json
    matched = json.loads(matched_json or "[]")
    missing = json.loads(missing_json or "[]")
    total = len(matched) + len(missing)
    keyword = (len(matched) / total) * 100 if total else 0
    weight = db.DEFAULT_KEYWORD_WEIGHT
    semantic = max(0.0, min(100.0, (score - weight * keyword) / (1 - weight)))
    return keyword, semantic


def get_or_create_demo_user(conn, email, password, name, company):
    password_hash = generate_password_hash(password)
    existing = conn.execute("SELECT id FROM users WHERE email = :email", {"email": email}).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE users SET full_name = :name, company_name = :company, password_hash = :hash
            WHERE id = :id
            """,
            {"name": name, "company": company, "hash": password_hash, "id": existing["id"]},
        )
        return existing["id"], False
    user_id = conn.execute(
        """
        INSERT INTO users (full_name, email, company_name, job_title, phone, password_hash, created_at)
        VALUES (:name, :email, :company, 'Demo account', '', :hash, :created_at)
        RETURNING id
        """,
        {
            "name": name, "email": email, "company": company, "hash": password_hash,
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        },
    ).scalar()
    return user_id, True


def wipe_user_jobs(conn, user_id):
    p = {"uid": user_id}
    conn.execute(
        """
        DELETE FROM resume_files WHERE candidate_id IN (
            SELECT cs.id FROM candidate_scores cs JOIN jobs j ON j.id = cs.job_id WHERE j.user_id = :uid)
        """, p)
    conn.execute(
        "DELETE FROM candidate_scores WHERE job_id IN (SELECT id FROM jobs WHERE user_id = :uid)", p)
    conn.execute("DELETE FROM activity WHERE job_id IN (SELECT id FROM jobs WHERE user_id = :uid)", p)
    conn.execute("DELETE FROM jobs WHERE user_id = :uid", p)


def table_columns(old, table):
    return {row[1] for row in old.execute(f"PRAGMA table_info({table})")}


def import_old_database(conn, user_id, source, uploads_dir):
    old = sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True)   # read-only: the old file is never modified
    old.row_factory = sqlite3.Row

    job_cols = table_columns(old, "jobs")
    cand_cols = table_columns(old, "candidate_scores")
    has_activity = bool(table_columns(old, "activity"))

    def pick(row, cols, name, default):
        return row[name] if name in cols and row[name] is not None else default

    job_map, cand_count, file_count, no_file = {}, 0, 0, []

    old_jobs = old.execute("SELECT * FROM jobs ORDER BY id").fetchall()
    for job in old_jobs:
        if pick(job, job_cols, "active", 1) != 1:
            continue   # archived in the old app: skip
        new_id = conn.execute(
            """
            INSERT INTO jobs (user_id, title, company, description, created_at, active, keyword_weight)
            VALUES (:uid, :title, :company, :description, :created_at, 1, :kw)
            RETURNING id
            """,
            {
                "uid": user_id, "title": job["title"], "company": job["company"],
                "description": job["description"], "created_at": job["created_at"],
                "kw": pick(job, job_cols, "keyword_weight", db.DEFAULT_KEYWORD_WEIGHT),
            },
        ).scalar()
        job_map[job["id"]] = new_id

    for cand in old.execute("SELECT * FROM candidate_scores ORDER BY id").fetchall():
        if cand["job_id"] not in job_map:
            continue
        keyword_score = pick(cand, cand_cols, "keyword_score", None)
        semantic_score = pick(cand, cand_cols, "semantic_score", None)
        if keyword_score is None or semantic_score is None:
            keyword_score, semantic_score = legacy_component_scores(
                cand["score"], cand["matched_skills"], cand["missing_skills"])
        resume_text = pick(cand, cand_cols, "resume_text", None)

        # Find the stored resume file first: it also lets us fill in missing resume text.
        file_data = None
        suffix = Path(cand["file_name"]).suffix.lower()
        if uploads_dir:
            path = find_uploaded_resume(uploads_dir, cand["file_name"], cand["created_at"])
            if path and suffix in CONTENT_TYPES and path.stat().st_size <= MAX_RESUME_BYTES:
                file_data = path.read_bytes()
        if resume_text is None and file_data:
            try:
                resume_text = extract_text(file_data, suffix)
            except Exception as e:
                print(f"  could not read text from {cand['file_name']}: {e}")

        new_cand_id = conn.execute(
            """
            INSERT INTO candidate_scores
                (job_id, candidate_name, file_name, score, matched_skills, missing_skills, created_at,
                 summary, keyword_score, semantic_score, resume_text)
            VALUES (:job_id, :name, :file_name, :score, :matched, :missing, :created_at,
                    :summary, :kw, :sem, :text)
            RETURNING id
            """,
            {
                "job_id": job_map[cand["job_id"]],
                "name": cand["candidate_name"],
                "file_name": cand["file_name"],
                "score": cand["score"],
                "matched": cand["matched_skills"],
                "missing": cand["missing_skills"],
                "created_at": cand["created_at"],
                "summary": pick(cand, cand_cols, "summary", ""),
                "kw": keyword_score,
                "sem": semantic_score,
                "text": resume_text,
            },
        ).scalar()
        cand_count += 1

        if file_data:
            conn.execute(
                """
                INSERT INTO resume_files (candidate_id, filename, content_type, size_bytes, data)
                VALUES (:cid, :filename, :ctype, :size, :data)
                """,
                {"cid": new_cand_id, "filename": cand["file_name"], "ctype": CONTENT_TYPES[suffix],
                 "size": len(file_data), "data": file_data},
            )
            file_count += 1
        elif uploads_dir:
            no_file.append(cand["file_name"])

    activity_count = 0
    if has_activity:
        for act in old.execute("SELECT * FROM activity ORDER BY id").fetchall():
            if act["job_id"] in job_map:
                conn.execute(
                    """
                    INSERT INTO activity (job_id, message, score, created_at)
                    VALUES (:job_id, :message, :score, :created_at)
                    """,
                    {"job_id": job_map[act["job_id"]], "message": act["message"],
                     "score": act["score"], "created_at": act["created_at"]},
                )
                activity_count += 1

    old.close()
    return len(job_map), cand_count, file_count, activity_count, no_file


def main():
    parser = argparse.ArgumentParser(description="Create the demo recruiter account and import old data.")
    parser.add_argument("--source", help="path to the old hiring_dashboard.db to import jobs/candidates from")
    parser.add_argument("--uploads", help="path to the old uploads/ folder, to attach resume files")
    parser.add_argument("--reset", action="store_true", help="delete the demo account's existing jobs first")
    args = parser.parse_args()

    email = os.getenv("DEMO_EMAIL", "").strip().lower()
    password = os.getenv("DEMO_PASSWORD", "")
    name = os.getenv("DEMO_NAME", "Demo Recruiter").strip() or "Demo Recruiter"
    company = os.getenv("DEMO_COMPANY", "TalentMatch Demo").strip() or "TalentMatch Demo"
    if not email or len(password) < 8:
        sys.exit("Set DEMO_EMAIL and DEMO_PASSWORD (at least 8 characters) in your .env file first.")
    if args.source and not os.path.isfile(args.source):
        sys.exit(f"Source database not found: {args.source}")
    if args.uploads and not os.path.isdir(args.uploads):
        sys.exit(f"Uploads folder not found: {args.uploads}")

    print("Database:", db.get_engine().url.render_as_string(hide_password=True))
    db.init_schema()

    with get_db() as conn:
        user_id, created = get_or_create_demo_user(conn, email, password, name, company)
        print(f"Demo account {'created' if created else 'updated'}: {email}")

        if not args.source:
            return

        existing = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE user_id = :uid", {"uid": user_id}
        ).fetchone()["n"]
        if existing and not args.reset:
            sys.exit(f"The demo account already has {existing} job(s). Nothing imported. "
                     "Re-run with --reset to replace them.")
        if args.reset:
            wipe_user_jobs(conn, user_id)
            print("Removed the demo account's previous jobs.")

        jobs, cands, files, acts, no_file = import_old_database(conn, user_id, args.source, args.uploads)

    print(f"Imported {jobs} job(s), {cands} candidate(s), {files} resume file(s), {acts} activity entries.")
    if cands and not args.uploads:
        print("No --uploads folder given, so candidates have no downloadable resume file.")
    elif no_file:
        print(f"{len(no_file)} candidate(s) have no matching file in the uploads folder (their scores were kept):")
        for file_name in no_file[:15]:
            print("  -", file_name)
    print("Reminder: this account is shared. Don't import real applicants' resumes into a public demo.")


if __name__ == "__main__":
    main()
