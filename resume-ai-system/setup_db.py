"""
One-time database setup. Creates the TalentMatch tables in whatever DATABASE_URL points to.

    python setup_db.py            # create any missing tables (safe to re-run)
    python setup_db.py --reset    # DELETE every table and its data, then recreate (asks first)

Run it once per database: once for your local/dev Neon project and once for the production one.
The web app does not create tables itself, so several server workers can't race each other.
"""
import sys

from dotenv import load_dotenv

load_dotenv()

import db  # noqa: E402  (after load_dotenv so DATABASE_URL from .env is visible)


def describe_target():
    url = db.get_engine().url
    # Never print the password.
    return url.render_as_string(hide_password=True)


def main():
    reset = "--reset" in sys.argv
    print(f"Database: {describe_target()}")

    if reset:
        answer = input("This will DELETE ALL DATA in that database. Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            print("Cancelled. Nothing was changed.")
            return
        db.drop_everything()
        print("Dropped all TalentMatch tables.")

    db.init_schema()
    print("Done. Tables ready:", ", ".join(sorted(t.name for t in db.metadata.sorted_tables)))


if __name__ == "__main__":
    main()
