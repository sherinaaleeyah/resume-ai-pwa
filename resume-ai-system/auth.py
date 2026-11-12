"""
Recruiter accounts: sign-up, login, logout, CSRF protection and session security.

Job seekers never sign in; only the /recruiter/... pages are protected.

Environment variables used here:
  SECRET_KEY              signs the login cookie. REQUIRED on Azure (the app refuses to start without it).
  RECRUITER_INVITE_CODE   if set, sign-up needs this code. If unset, anyone can create a recruiter account.
"""
import hmac
import os
import re
from datetime import datetime
from urllib.parse import urlparse

from flask import Blueprint, redirect, render_template, request, url_for
from flask_login import (
    LoginManager, UserMixin, current_user, login_user, logout_user,
)
from flask_wtf.csrf import CSRFError, CSRFProtect
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from db import get_db, running_on_azure

bp = Blueprint("auth", __name__)
login_manager = LoginManager()
csrf = CSRFProtect()

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8

# Hashed once at import. Used so a login attempt for an unknown email takes as long as one
# for a real email (otherwise response time reveals which emails have accounts).
_DUMMY_HASH = generate_password_hash("not-a-real-password")


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------
class User(UserMixin):
    def __init__(self, row):
        self.id = row["id"]
        self.full_name = row["full_name"]
        self.email = row["email"]
        self.company_name = row["company_name"]
        self.job_title = row["job_title"]
        self.phone = row["phone"]

    def get_id(self):
        return str(self.id)


def _get_user_by_id(user_id):
    with get_db() as conn:
        return conn.execute("SELECT * FROM users WHERE id = :v", {"v": user_id}).fetchone()


def _get_user_by_email(email):
    with get_db() as conn:
        return conn.execute("SELECT * FROM users WHERE email = :v", {"v": email}).fetchone()


@login_manager.user_loader
def load_user(user_id):
    try:
        row = _get_user_by_id(int(user_id))
    except (TypeError, ValueError):
        return None
    return User(row) if row else None


def invite_code_required():
    return bool(os.getenv("RECRUITER_INVITE_CODE", "").strip())


def _invite_code_ok(submitted):
    expected = os.getenv("RECRUITER_INVITE_CODE", "").strip()
    if not expected:
        return True
    # Constant-time comparison so the code can't be guessed character by character.
    return hmac.compare_digest(submitted.strip().encode("utf-8"), expected.encode("utf-8"))


def _safe_next_url(target):
    """Only follow `next` redirects that stay on this site."""
    if not target:
        return None
    parts = urlparse(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return None
    return target


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------
def init_app(app):
    secret = os.getenv("SECRET_KEY", "").strip()
    if not secret:
        if running_on_azure():
            raise RuntimeError(
                "SECRET_KEY is not set. Add it under App Service > Environment variables "
                "(any long random string), then restart the app."
            )
        secret = "dev-only-insecure-key"   # local development only
    app.config["SECRET_KEY"] = secret

    secure = running_on_azure()          # Azure serves HTTPS; localhost is plain HTTP
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=secure,
        REMEMBER_COOKIE_HTTPONLY=True,
        REMEMBER_COOKIE_SAMESITE="Lax",
        REMEMBER_COOKIE_SECURE=secure,
        WTF_CSRF_TIME_LIMIT=None,        # a token is valid as long as the login session
    )

    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to use the recruiter tools."
    login_manager.login_message_category = "info"
    login_manager.init_app(app)
    csrf.init_app(app)
    app.register_blueprint(bp)

    @app.before_request
    def _protect_recruiter_area():
        # Safety net on top of @login_required: anything under /recruiter needs a login,
        # even a route added in future by someone who forgets the decorator.
        if request.path.startswith("/recruiter") and not current_user.is_authenticated:
            return login_manager.unauthorized()

    @app.errorhandler(CSRFError)
    def _csrf_error(error):
        return render_template(
            "error.html",
            title="Page expired",
            message="Your form session expired or was invalid. Go back, refresh the page and try again.",
        ), 400

    @app.errorhandler(413)
    def _too_large(error):
        return render_template(
            "error.html",
            title="Upload too large",
            message="That upload is too large. Each resume must be under 5 MB, and a single upload under 50 MB in total.",
        ), 413

    @app.after_request
    def _no_store_for_private_pages(response):
        # Logged-in pages must not be kept by browsers/proxies, or Back after logout would show them.
        if request.path.startswith("/recruiter") or request.path in ("/login", "/signup"):
            response.headers["Cache-Control"] = "no-store"
        return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("recruiter"))

    form = {"full_name": "", "email": "", "company_name": "", "job_title": "", "phone": ""}
    error = None

    if request.method == "POST":
        for key in form:
            form[key] = request.form.get(key, "").strip()
        form["email"] = form["email"].lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        invite = request.form.get("invite_code", "")

        if not form["full_name"] or not form["email"] or not form["company_name"] or not password:
            error = "Please fill in your name, work email, company name and a password."
        elif not EMAIL_RE.match(form["email"]) or len(form["email"]) > 254:
            error = "Please enter a valid email address."
        elif len(form["full_name"]) > 100 or len(form["company_name"]) > 120 \
                or len(form["job_title"]) > 100 or len(form["phone"]) > 30:
            error = "One of the fields is too long."
        elif len(password) < MIN_PASSWORD_LENGTH:
            error = f"Your password must be at least {MIN_PASSWORD_LENGTH} characters."
        elif len(password) > 128:
            error = "Your password is too long (128 characters maximum)."
        elif password != confirm:
            error = "The two passwords don't match."
        elif not _invite_code_ok(invite):
            error = "That invite code is not valid. Ask the TalentMatch team for the code."
        else:
            try:
                with get_db() as conn:
                    user_id = conn.execute(
                        """
                        INSERT INTO users (full_name, email, company_name, job_title, phone, password_hash, created_at)
                        VALUES (:full_name, :email, :company_name, :job_title, :phone, :password_hash, :created_at)
                        RETURNING id
                        """,
                        {
                            **form,
                            "password_hash": generate_password_hash(password),
                            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
                        },
                    ).scalar()
            except IntegrityError:
                error = "An account with that email already exists. Try signing in instead."
            else:
                row = _get_user_by_id(user_id)
                login_user(User(row))
                return redirect(url_for("recruiter"))

    return render_template("signup.html", form=form, error=error, invite_required=invite_code_required()), \
        (400 if error else 200)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("recruiter"))

    error = None
    email = ""
    next_url = _safe_next_url(request.values.get("next"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        row = _get_user_by_email(email) if email else None

        # Same message and similar timing whether the email or the password was wrong.
        password_ok = check_password_hash(row["password_hash"] if row else _DUMMY_HASH, password)
        if row and password_ok:
            login_user(User(row))
            return redirect(next_url or url_for("recruiter"))
        error = "Incorrect email or password."

    return render_template("login.html", email=email, error=error, next_url=next_url), (401 if error else 200)


@bp.route("/logout", methods=["POST"])
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
