from datetime import datetime
from huggingface_hub import list_repo_refs

from .constants import LOG_FILE

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as fout:
        date = datetime.now().strftime("%d.%m.%Y %H:%M")
        fout.write(f"[{date}] {msg}\n")

def sort_versions(versions):
    def version_key(v):
        return [int(x) for x in v.split('.')]
    
    return sorted(versions, key=version_key)

def get_ds_versions(repo_id):
    repo_refs = list_repo_refs(repo_id, repo_type="dataset")
    return [ref.name for ref in repo_refs.tags]

def get_latest_version(versions):
    return sort_versions(versions)[-1]

