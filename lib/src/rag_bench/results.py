import json
import os
import uuid
from pathlib import Path

import requests


def save(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(path, "w", encoding="utf8"), ensure_ascii=False, indent=4)


def load(path):
    return json.load(open(path, "r", encoding="utf8"))


def submit(path, config, address="localhost:80", is_auto=False, submit_id="", version="", token=""):
    with open(path, "r", encoding="utf-8") as file:
        content = json.load(file)
    # TODO do some content validation before sending

    config["user_guid"] = uuid.uuid4().hex
    config["is_auto"] = is_auto

    with open(path, "rb") as file:
        data = {
            "config": json.dumps(config),
            "submit_id": submit_id,
            "version": version,
            "token": token,
            "filename": os.path.basename(path),
        }
        files = {path: file}
        response = requests.post(
            f"http://{address}/submit/create",
            data=data,
            files=files,
        )

    try:
        res = json.loads(response.content.decode("utf-8"))
    except Exception as e:
        print("Exception occured:", str(e))
        return

    print(f"Done. Your submission id is {res['id']}")
    return res['id']
