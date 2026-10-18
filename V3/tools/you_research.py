"""Dataset hunt with the You.com Research API (structured output).

Key: SEARCH_YDC_API_KEY (or YDC_API_KEY) from the environment, V3/.env or V2/.env
(read-only). Never printed.

    POST https://api.you.com/v1/research
    headers  X-API-Key
    body     {input, research_effort, output_schema}

Usage:  python you_research.py [--effort deep]
Writes: V3/docs/dataset_search/you_research_<effort>_<date>.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

V3 = Path(__file__).resolve().parents[1]
OUT = V3 / "docs" / "dataset_search"
ENV_FILES = [V3 / ".env", V3.parent / "V2" / ".env"]
ENDPOINT = "https://api.you.com/v1/research"


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------

def api_key() -> str:
    for name in ("SEARCH_YDC_API_KEY", "YDC_API_KEY"):
        if os.environ.get(name):
            return os.environ[name].strip()
    for env in ENV_FILES:
        if not env.exists():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("SEARCH_YDC_API_KEY", "YDC_API_KEY") and v.strip():
                return v.strip().strip('"').strip("'")
    raise SystemExit("no You.com key found (SEARCH_YDC_API_KEY in V3/.env or V2/.env)")


# ---------------------------------------------------------------------------
# Questions and output schema
# ---------------------------------------------------------------------------

QUESTIONS = {
    "esp32_counting": "Find every WiFi CSI dataset collected with ESP32 boards (any variant: ESP32, S2, S3, C3, C5, C6) "
                      "that has people-count or room-occupancy labels (number of people, 0/1/2+, crowd size). "
                      "Include datasets released on GitHub, Zenodo, Kaggle, Hugging Face, IEEE DataPort, figshare, "
                      "Mendeley Data, or available on request from paper authors (2020-2026).",
    "esp32_real_noise": "Find ESP32 WiFi CSI datasets recorded in real, uncontrolled environments with natural noise: "
                        "homes, offices, kitchens, classrooms, factories; running fans, air conditioners, HVAC, "
                        "appliances or machines; long-term multi-day recordings; crowded 2.4 GHz WiFi channels.",
    "nonhuman_motion": "Find WiFi CSI datasets (any hardware) that label non-human motion or interference separately "
                       "from people: fans, ceiling fans, air conditioners, pets, robot vacuums, curtains, machines.",
    "crowd_counting_any": "Find publicly downloadable WiFi CSI crowd counting / people counting datasets (any hardware, "
                          "2019-2026) recorded across multiple rooms or days, with counts above 3 people.",
    "interference_traffic": "Find WiFi sensing CSI datasets that record co-channel interference, network traffic load, "
                            "packet loss or irregular sampling as a variable (for example ESP32 or commodity routers "
                            "sensing while other WiFi networks are busy).",
    "esp32_presence_multiroom": "Find ESP32 WiFi CSI datasets for presence / empty-room detection recorded in several "
                                "different rooms or buildings (cross-environment), with download links.",
}

NULL_STR = {"type": ["string", "null"]}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["datasets"],
    "properties": {
        "datasets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "url", "hardware", "is_esp32", "labels", "people_count_range",
                             "environments", "noise_conditions", "size", "access", "license", "paper"],
                "properties": {
                    "name": {"type": "string"},
                    "url": NULL_STR,
                    "hardware": NULL_STR,
                    "is_esp32": {"type": ["boolean", "null"]},
                    "labels": NULL_STR,
                    "people_count_range": NULL_STR,
                    "environments": NULL_STR,
                    "noise_conditions": NULL_STR,
                    "size": NULL_STR,
                    "access": NULL_STR,
                    "license": NULL_STR,
                    "paper": NULL_STR,
                },
            },
        }
    },
}


def ask(key: str, topic: str, question: str, effort: str) -> dict:
    t = time.time()
    body = {"input": question, "research_effort": effort, "output_schema": SCHEMA}
    r = httpx.post(ENDPOINT, json=body, headers={"X-API-Key": key, "Content-Type": "application/json"},
                   timeout=400)
    took = time.time() - t
    if r.status_code != 200:
        return dict(topic=topic, ok=False, status=r.status_code, error=r.text[:400], seconds=took)
    data = r.json()
    out = data.get("output", {})
    return dict(topic=topic, ok=True, seconds=took, content=out.get("content"),
                sources=out.get("sources", []), warnings=data.get("warnings", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", default="deep", choices=["lite", "standard", "deep", "exhaustive"])
    ap.add_argument("--topics", nargs="+", default=list(QUESTIONS), choices=list(QUESTIONS))
    args = ap.parse_args()
    questions = {t: QUESTIONS[t] for t in args.topics}

    key = api_key()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"running {len(questions)} research questions ({args.effort}) ...", flush=True)

    with ThreadPoolExecutor(len(questions)) as pool:
        futs = {t: pool.submit(ask, key, t, q, args.effort) for t, q in questions.items()}
        results = []
        for t, f in futs.items():
            res = f.result()
            results.append(res)
            n = len((res.get("content") or {}).get("datasets", [])) if res.get("ok") else 0
            print(f"  {t:26s} {'ok' if res['ok'] else 'FAILED ' + str(res.get('status'))}  "
                  f"{n} datasets  {len(res.get('sources', []))} sources  {res['seconds']:.0f}s", flush=True)

    stamp = time.strftime("%Y-%m-%d_%H%M")
    path = OUT / f"you_research_{args.effort}_{stamp}.json"
    path.write_text(json.dumps(dict(effort=args.effort, questions=questions, results=results), indent=1),
                    encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
