import json
import os
from pathlib import Path

from PIL import Image
from loguru import logger as eval_logger

AMBER_ROOT = os.environ.get("AMBER_ROOT", "/root/autodl-tmp/Method/dataset/AMBER")
AMBER_IMAGE_DIR = os.environ.get("AMBER_IMAGE_DIR", os.path.join(AMBER_ROOT, "image"))
GENERATIVE_MAX_ID = 1004


def _image_path(doc):
    name = doc["image"]
    if os.path.isabs(name):
        return name
    return os.path.join(AMBER_IMAGE_DIR, name)


def amber_doc_to_visual(doc):
    path = _image_path(doc)
    return [Image.open(path).convert("RGB")]


def amber_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    query = str(doc["query"]).strip()
    sample_id = int(doc["id"])
    if sample_id > GENERATIVE_MAX_ID and not post_prompt:
        post_prompt = " Answer with Yes or No."
    return f"{pre_prompt}{query}{post_prompt}"


def _normalize_discriminative(text):
    """AMBER scoring requires the exact strings 'Yes' or 'No'."""
    stripped = (text or "").strip()
    if not stripped:
        return stripped
    first = stripped.split()[0].strip(".,!?;:\"'`").lower()
    if first.startswith("yes"):
        return "Yes"
    if first.startswith("no"):
        return "No"
    lower = stripped.lower()
    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"
    return stripped


def amber_process_results(doc, results):
    pred = results[0] if results else ""
    sample_id = int(doc["id"])
    if sample_id > GENERATIVE_MAX_ID:
        pred = _normalize_discriminative(pred)
    return {
        "amber_responses": {
            "id": sample_id,
            "response": pred,
        }
    }


def amber_aggregate_responses(results, args=None):
    results = sorted(results, key=lambda x: int(x["id"]))
    payload = [{"id": int(r["id"]), "response": r["response"]} for r in results]

    method = os.getenv("METHOD", "shiftkv")
    budget = os.getenv("BUDGET", "64")
    ratio = os.getenv("RATIO", "0.1")
    default_name = f"amber_{method}_{budget}_{ratio}.json"

    out_path = os.environ.get("AMBER_OUTPUT")
    if not out_path:
        if args is not None and getattr(args, "output_path", None):
            out_dir = os.path.join(args.output_path, "amber_responses")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, default_name)
        else:
            project = os.getenv("PROJECT", "MM-ShiftKV")
            model_name = os.getenv("MODEL_NAME", "unknown_model")
            out_dir = os.path.join(AMBER_ROOT, "results", project, model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, default_name)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    eval_logger.info(f"Wrote {len(payload)} AMBER responses to {out_path}")
    os.environ["AMBER_LAST_OUTPUT"] = os.path.abspath(out_path)
    return len(payload)
