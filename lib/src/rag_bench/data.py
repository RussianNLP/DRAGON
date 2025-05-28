from datasets import load_dataset
from .helper import log, get_latest_version, get_ds_versions
from .constants import PUBLIC_TEXTS_REPO_ID, PUBLIC_QUESTIONS_REPO_ID


def get_texts():
    log("Loading texts dataset")
    return load_dataset(PUBLIC_TEXTS_REPO_ID)

def get_questions():
    log("Loading questions dataset")
    return load_dataset(PUBLIC_QUESTIONS_REPO_ID)

def get_datasets():
    text_versions = get_ds_versions(PUBLIC_TEXTS_REPO_ID)
    question_versions = get_ds_versions(PUBLIC_QUESTIONS_REPO_ID)

    latest_text_version = get_latest_version(text_versions)
    latest_question_version = get_latest_version(question_versions)
    
    log(f"Latest texts version: {latest_text_version}")
    log(f"Latest questions version: {latest_question_version}")

    assert latest_text_version == latest_question_version, "Texts and questions datasets have different versions"

    text_ds = load_dataset(PUBLIC_TEXTS_REPO_ID, revision=latest_text_version)
    question_ds = load_dataset(PUBLIC_QUESTIONS_REPO_ID, revision=latest_question_version)

    log(f"Loaded texts dataset with {len(text_ds['train'])} texts")
    log(f"Loaded questions dataset with {len(question_ds['train'])} questions")

    return text_ds, question_ds, latest_text_version
