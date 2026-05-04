import re
import spacy

# Load spaCy model (make sure you've run: python -m spacy download en_core_web_sm)
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
