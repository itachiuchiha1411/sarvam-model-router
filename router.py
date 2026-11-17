"""
router.py - a small model router for the Sarvam GTM assignment.

It reads test prompts, decides which of three models should answer each one,
calls that model, scores the answer with a judge model, and writes a results
spreadsheet plus a summary table. All settings and prices are in config.py.

Commands (run them in a terminal, one at a time):

  python router.py --check                                     check that every model answers
  python router.py --classify-test                             show how 5 sample prompts would be routed
  python router.py --config router --set practice --no-judge   route the 10 practice prompts
  python router.py --judge --config router --set practice      score the practice answers
  python router.py --config always_cheapest --set test         baseline 1: every test prompt to Tier 1
  python router.py --config always_frontier --set test         baseline 2: every test prompt to Tier 3
  python router.py --config router --set test                  the router on the 40 test prompts
  python router.py --summary                                   write and print the results table
  python router.py --fallback-test                             force a fallback on 3 practice prompts
  python router.py --probe                                     try many free models once to see which work
  python router.py --list-models [WORD] --provider nvidia      list a provider's model names
  python router.py --spend                                     show the cost of all live API calls so far

Add --limit 3 to any --config run to try only the first 3 prompts.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from datetime import datetime

import httpx
import openai
import pandas as pd
from openai import OpenAI

import config

# Make Hindi, Tamil and other scripts print correctly in the Windows terminal.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# The three answering tiers, cheapest first. "Next tier up" means the next one in this list.
TIERS = ["tier1", "tier2", "tier3"]
CONFIGS = ["always_cheapest", "always_frontier", "router"]


# ---------------------------------------------------------------------------
# A special error that stops the whole run: a missing or rejected key, no credits,
# or a wrong model name. Falling back to another model would hide these problems.
# ---------------------------------------------------------------------------
class StopRun(Exception):
    pass


# ---------------------------------------------------------------------------
# Connections to each provider. Created once and reused.
# max_retries=0 switches off the library's own silent retries, so every wait
# and retry is ours and gets logged.
# ---------------------------------------------------------------------------
_clients = {}


def api_key_for(provider):
    env_name = config.PROVIDERS[provider]["key_env"]
    key = os.getenv(env_name, "").strip().strip('"').strip("'")
    if not key:
        raise StopRun(f"{env_name} is missing or empty in your .env file.")
    return key


def get_client(provider):
    if provider not in _clients:
        key = api_key_for(provider)
        # Sarvam's docs use an "api-subscription-key" header; we send it as well as the standard one.
        headers = {"api-subscription-key": key} if provider == "sarvam" else None
        _clients[provider] = OpenAI(api_key=key, base_url=config.PROVIDERS[provider]["base_url"],
                                    default_headers=headers, max_retries=0)
    return _clients[provider]


# ---------------------------------------------------------------------------
# Cost: tokens / 1,000,000 x price per million, converted to rupees if the price is in dollars.
# ---------------------------------------------------------------------------
def cost_inr(model_key, input_tokens, output_tokens):
    m = config.MODELS[model_key]
    rate = config.USD_TO_INR if m["currency"] == "USD" else 1.0
    return (input_tokens / 1_000_000 * m["price_in"] + output_tokens / 1_000_000 * m["price_out"]) * rate


# ---------------------------------------------------------------------------
# A log of every live API call (not cache reuse), so spend can be totalled at any time.
# ---------------------------------------------------------------------------
def log_api_call(model_key, input_tokens, output_tokens, cost, outcome):
    is_new = not os.path.exists(config.CALL_LOG_FILE)
    m = config.MODELS[model_key]
    with open(config.CALL_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["time", "role", "model", "provider", "input_tokens", "output_tokens", "cost_inr", "outcome"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), model_key, m["model"], m["provider"],
                         input_tokens, output_tokens, round(cost, 6), outcome])


def uses_sarvam():
    in_tiers = any(config.MODELS[k]["provider"] == "sarvam" for k in ("tier1", "tier2", "tier3", "judge"))
    return in_tiers or config.CLASSIFIER == "prism"


def print_spend():
    """Show the list-price value of every live call so far. Free tiers mean nothing is actually paid."""
    if not os.path.exists(config.CALL_LOG_FILE):
        return
    log = pd.read_csv(config.CALL_LOG_FILE, encoding="utf-8")
    total = float(log["cost_inr"].sum())
    print(f"\nList-price cost of all live calls so far: Rs {total:.2f}. "
          f"These calls run on free tiers, so nothing is actually charged.")
    if uses_sarvam():
        used = float(log.loc[log["provider"] == "sarvam", "cost_inr"].sum())
        print(f"Sarvam credit used so far (estimate): Rs {used:.2f} of Rs {config.SARVAM_FREE_CREDIT_INR:.0f}. "
              f"Check the exact balance on the Sarvam dashboard.")


# ---------------------------------------------------------------------------
# The cache: a file of saved answers. Before calling a model we look here for an earlier
# successful answer from the same model to the same prompt, and reuse it (with its original
# tokens, cost and latency) instead of paying again.
# ---------------------------------------------------------------------------
_caches = {}


def cache_key(model_name, text):
    return hashlib.sha256(f"{model_name}||{text}".encode("utf-8")).hexdigest()


def load_cache(path):
    if path not in _caches:
        store = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        store[record["key"]] = record
                    except Exception:
                        pass  # skip a damaged line rather than crash
        _caches[path] = store
    return _caches[path]


def save_to_cache(path, key, result):
    record = {"key": key, "saved": datetime.now().isoformat(timespec="seconds"), "result": result}
    load_cache(path)[key] = record
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------
def clean_answer(text):
    """Remove any <think>...</think> block some models put in the answer, and trim spaces."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def is_refusal(answer):
    return answer.strip().lower().startswith(config.REFUSAL_STARTS)


def normalise(text):
    """Lower-case and drop spaces, punctuation and symbols, so '₹12,500.' matches '12500'."""
    kept = [ch for ch in str(text).lower()
            if unicodedata.category(ch)[0] not in ("P", "Z", "S", "C")]
    return "".join(kept)


def short(text, n=160):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


# ---------------------------------------------------------------------------
# One request to one model.
# - On "too many requests" (429): wait and retry the same model, up to RATE_LIMIT_RETRIES times.
# - On a timeout or other error: return the problem so the caller can fall back.
# - Latency is only the time waiting for the reply, never the waits.
# ---------------------------------------------------------------------------
def call_model(model_key, messages, timeout=None, read_cache=True, write_cache=True, cache_text=None,
               rate_limit_wait=None, retry_when_busy=False):
    m = config.MODELS[model_key]
    text_for_key = cache_text if cache_text is not None else json.dumps(messages, ensure_ascii=False)
    key = cache_key(m["model"], text_for_key)

    # 1. Reuse a saved answer if we have one.
    if read_cache:
        hit = load_cache(config.CACHE_FILE).get(key)
        if hit:
            return dict(hit["result"], from_cache=True, rate_limit_retries=0)

    # 2. Otherwise call the model.
    client = get_client(m["provider"])
    params = dict(model=m["model"], messages=messages, max_tokens=config.MAX_TOKENS,
                  timeout=timeout or config.TIMEOUT_SECONDS)
    if m.get("reasoning_effort"):
        params["reasoning_effort"] = m["reasoning_effort"]
    wait = rate_limit_wait or config.RATE_LIMIT_WAIT_SECONDS
    waits = 0

    def failed(status, message, latency_ms=0.0):
        return {"status": status, "error": message, "answer": "", "input_tokens": 0, "output_tokens": 0,
                "reasoning_tokens": 0, "cost_inr": 0.0, "latency_ms": round(latency_ms, 1),
                "finish_reason": "", "from_cache": False, "rate_limit_retries": waits}

    while True:
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(**params)
            latency_ms = (time.perf_counter() - start) * 1000
            break
        except openai.RateLimitError as e:
            if waits >= config.RATE_LIMIT_RETRIES:
                log_api_call(model_key, 0, 0, 0.0, "rate_limited")
                return failed("rate_limited", f"429 after {waits} waits: {short(e, 200)}")
            waits += 1
            print(f"      {m['label']} says too many requests; waiting {wait}s (retry {waits} of {config.RATE_LIMIT_RETRIES})")
            time.sleep(wait)
        except openai.APITimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            log_api_call(model_key, 0, 0, 0.0, "timeout")
            return failed("timeout", f"no reply within {params['timeout']}s", latency_ms)
        except openai.APIStatusError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            # "Busy" errors (500, 502, 503: the provider's servers are overloaded) are temporary.
            # For the judge we wait and retry, because a measurement should not depend on a busy moment.
            # For answering models we do not: falling back to another model is the router's job.
            if retry_when_busy and e.status_code in (500, 502, 503) and waits < config.RATE_LIMIT_RETRIES:
                waits += 1
                print(f"      {m['label']} is busy ({e.status_code}); waiting {wait}s "
                      f"(retry {waits} of {config.RATE_LIMIT_RETRIES})")
                time.sleep(wait)
                continue
            log_api_call(model_key, 0, 0, 0.0, f"error {e.status_code}")
            env_name = config.PROVIDERS[m["provider"]]["key_env"]
            if e.status_code in (401, 403):
                raise StopRun(f"{m['label']}: access refused ({e.status_code}). Check {env_name} in .env; "
                              f"if the key is right, this model may not be open to your account.")
            if e.status_code == 402:
                raise StopRun(f"{m['label']}: out of credits (402). Check your balance on the provider's dashboard.")
            if e.status_code == 404:
                raise StopRun(f"{m['label']}: model '{m['model']}' not found (404). Check the model name in config.py.")
            return failed("error", f"{e.status_code}: {short(e, 300)}", latency_ms)
        except openai.APIConnectionError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            log_api_call(model_key, 0, 0, 0.0, "connection error")
            return failed("error", f"connection error: {short(e, 200)}", latency_ms)

    # 3. Read the answer and the token counts the model reports (never estimated).
    choice = response.choices[0] if response.choices else None
    answer = clean_answer(choice.message.content if choice and choice.message else "")
    usage = response.usage
    input_tokens = (usage.prompt_tokens or 0) if usage else 0
    completion = (usage.completion_tokens or 0) if usage else 0
    total = (usage.total_tokens or 0) if usage else 0
    # Thinking tokens are billed as output. Some providers count them inside completion_tokens,
    # others only inside total_tokens, so we take whichever is larger.
    output_tokens = max(completion, total - input_tokens)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
    cost = cost_inr(model_key, input_tokens, output_tokens)
    log_api_call(model_key, input_tokens, output_tokens, cost, "ok")

    result = {"status": "ok", "error": "" if usage else "provider sent no token counts",
              "answer": answer, "input_tokens": input_tokens, "output_tokens": output_tokens,
              "reasoning_tokens": reasoning_tokens, "cost_inr": cost, "latency_ms": round(latency_ms, 1),
              "finish_reason": (choice.finish_reason or "") if choice else ""}

    # 4. Save successful, non-empty answers so a rerun never pays twice.
    if write_cache and answer:
        save_to_cache(config.CACHE_FILE, key, result)

    # 5. Pause before the next live call, to stay under free-tier limits.
    time.sleep(provider_pause(m["provider"]))
    return dict(result, from_cache=False, rate_limit_retries=waits)


def provider_pause(provider):
    return config.PAUSE_BY_PROVIDER.get(provider, config.PAUSE_BETWEEN_CALLS_SECONDS)


# ---------------------------------------------------------------------------
# The classifier. Sarvam Prism answers two multiple-choice questions about each prompt and
# returns a probability for every option. "Language" also counts the language the answer must
# be written in, so "Translate this into Marathi" is treated as an Indian-language task.
# ---------------------------------------------------------------------------
LANGUAGE_QUESTION = {
    "type": "choice",
    "instructions": "Which language situation applies to this task? Count both the language the message is "
                    "written in and the language the answer must be written in.",
    "criteria": {
        "english": "English only: written in English and the answer should be in English.",
        "indian_language": "Mainly an Indian language (such as Hindi, Tamil, Bengali or Marathi), in its own "
                           "script or romanised, OR the task asks for an answer or translation in an Indian language.",
        "code_mixed": "Mixes English with an Indian language in the same message, for example Hinglish.",
    },
}
DIFFICULTY_QUESTION = {
    "type": "choice",
    "instructions": "How hard is this task for an AI model to do correctly?",
    "criteria": {
        "simple": "A short single-step task: classify, extract one fact, answer yes or no, or write a one-line reply.",
        "moderate": "A standard task needing a few steps: summarise a conversation or document, draft a reply, "
                    "translate a passage, or answer a question from a document.",
        "complex": "Multi-step reasoning: comparing options, calculating across several facts, or making a "
                   "recommendation that needs justification.",
    },
}
LANGUAGE_LABELS = {"english": "English", "indian_language": "Indian language", "code_mixed": "Code-mixed"}


def classify_prism(text, read_cache=True, write_cache=True, timeout=None):
    """Ask Sarvam Prism the two questions. Returns a dict with ok=True, or ok=False and the problem."""
    key = cache_key("sarvam-prism", text)
    if read_cache:
        hit = load_cache(config.CACHE_FILE).get(key)
        if hit:
            return dict(hit["result"], from_cache=True, rate_limit_retries=0)

    api_key = api_key_for("sarvam")
    body = {"model": config.MODELS["router"]["model"], "state": text,
            "questions": {"language": LANGUAGE_QUESTION, "difficulty": DIFFICULTY_QUESTION}}
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    waits = 0
    while True:
        start = time.perf_counter()
        try:
            response = httpx.post(config.PRISM_URL, json=body, headers=headers,
                                  timeout=timeout or config.TIMEOUT_SECONDS)
        except httpx.TimeoutException:
            log_api_call("router", 0, 0, 0.0, "timeout")
            return {"ok": False, "problem": "timeout", "latency_ms": (time.perf_counter() - start) * 1000,
                    "rate_limit_retries": waits}
        except httpx.HTTPError as e:
            log_api_call("router", 0, 0, 0.0, "connection error")
            return {"ok": False, "problem": f"connection error: {short(e, 120)}",
                    "latency_ms": (time.perf_counter() - start) * 1000, "rate_limit_retries": waits}
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 429 and waits < config.RATE_LIMIT_RETRIES:
            waits += 1
            print(f"      Prism says too many requests; waiting {config.RATE_LIMIT_WAIT_SECONDS}s "
                  f"(retry {waits} of {config.RATE_LIMIT_RETRIES})")
            time.sleep(config.RATE_LIMIT_WAIT_SECONDS)
            continue
        break

    if response.status_code != 200:
        log_api_call("router", 0, 0, 0.0, f"error {response.status_code}")
        if response.status_code in (401, 403):
            raise StopRun(f"Sarvam Prism: the API key was rejected ({response.status_code}). Check SARVAM_API_KEY in .env.")
        if response.status_code == 402:
            raise StopRun("Sarvam Prism: out of credits (402). Check your balance on the Sarvam dashboard.")
        return {"ok": False, "problem": f"error {response.status_code}: {short(response.text, 200)}",
                "latency_ms": latency_ms, "rate_limit_retries": waits}

    try:
        data = response.json()
        lang = data["answers"]["language"]
        diff = data["answers"]["difficulty"]
        lang["probabilities"][lang["choice"]], diff["probabilities"][diff["choice"]]
    except Exception:
        log_api_call("router", 0, 0, 0.0, "unexpected reply")
        return {"ok": False, "problem": f"unexpected reply: {short(response.text, 200)}",
                "latency_ms": latency_ms, "rate_limit_retries": waits}
    tokens = int((data.get("usage") or {}).get("input_tokens", 0) or 0)
    cost = cost_inr("router", tokens, 0)
    log_api_call("router", tokens, 0, cost, "ok")
    result = {"ok": True, "classifier": "prism",
              "language": lang["choice"], "language_prob": float(lang["probabilities"][lang["choice"]]),
              "difficulty": diff["choice"], "difficulty_prob": float(diff["probabilities"][diff["choice"]]),
              "difficulty_confidence": float(diff.get("confidence", 0) or 0),
              "tokens": tokens, "cost_inr": cost, "latency_ms": round(latency_ms, 1), "note": ""}
    if write_cache:
        save_to_cache(config.CACHE_FILE, key, result)
    time.sleep(config.PAUSE_BETWEEN_CALLS_SECONDS)
    return dict(result, from_cache=False, rate_limit_retries=waits)


# The backup classifier: simple rules, no API call, no cost.
INDIC_SCRIPT_RANGES = [(0x0900, 0x0D7F)]  # Devanagari through Malayalam: Hindi, Marathi, Bengali, Tamil, etc.
HINGLISH_WORDS = {"hai", "hain", "nahi", "nahin", "kya", "mera", "meri", "mujhe", "aap", "aapka", "kar", "karo",
                  "raha", "rahi", "gaya", "gayi", "kyun", "kyon", "paisa", "paise", "bhai", "yaar", "abhi", "kab",
                  "kaise", "haan", "theek", "tha", "thi", "chahiye", "wala", "wali", "kuch", "bahut", "jaldi",
                  "kitna", "kripya", "hoga", "diya", "liya", "mein", "se", "ko", "ka", "ki", "ke"}
INDIAN_LANGUAGE_NAMES = {"hindi", "marathi", "tamil", "telugu", "bengali", "bangla", "kannada", "malayalam",
                         "gujarati", "punjabi", "odia", "oriya", "assamese", "urdu"}
COMPLEX_WORDS = ("compare", "comparison", "analyse", "analyze", "calculate", "recommend", "evaluate",
                 "which is better", "pros and cons", "step by step", "reconcile", "justify", "trade-off")
# Note: "reply" and "write" are deliberately NOT here. On the practice prompts (P01, P02), instructions like
# "Reply with the category only" made simple classification look like a drafting task.
MODERATE_WORDS = ("summarise", "summarize", "summary", "translate", "draft", "explain")


def classify_rules(text):
    start = time.perf_counter()
    letters = [ch for ch in text if ch.isalpha()]
    indic = [ch for ch in letters if any(lo <= ord(ch) <= hi for lo, hi in INDIC_SCRIPT_RANGES)]
    indic_share = len(indic) / len(letters) if letters else 0.0
    words = re.findall(r"[a-zA-Z]+", text.lower())
    hinglish_hits = sum(1 for w in words if w in HINGLISH_WORDS)
    lower = text.lower()

    # Language
    if indic_share > 0.5:
        language = "indian_language"
    elif indic_share > 0.05 or hinglish_hits >= 3:
        language = "code_mixed"
    elif any(name in words for name in INDIAN_LANGUAGE_NAMES) and ("translate" in lower or " in " in lower):
        language = "indian_language"  # e.g. "Translate this SMS into Marathi"
    else:
        language = "english"

    # Difficulty
    word_count = len(text.split())
    if any(w in lower for w in COMPLEX_WORDS):
        difficulty = "complex"
    elif word_count > 80 or any(w in lower for w in MODERATE_WORDS):
        difficulty = "moderate"
    else:
        difficulty = "simple"

    return {"ok": True, "classifier": "rules", "language": language, "language_prob": 1.0,
            "difficulty": difficulty, "difficulty_prob": 1.0, "difficulty_confidence": 1.0,
            "tokens": 0, "cost_inr": 0.0, "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "note": "rules (no confidence score)", "from_cache": False, "rate_limit_retries": 0}


def classify(text, read_cache=True, write_cache=True, prism_timeout=None):
    """Use the classifier set in config.py. If Prism fails on this prompt, use the rules for it."""
    if config.CLASSIFIER == "rules":
        return classify_rules(text)
    result = classify_prism(text, read_cache=read_cache, write_cache=write_cache, timeout=prism_timeout)
    if result["ok"]:
        return result
    backup = classify_rules(text)
    # The customer still waited for the failed Prism call, so its time counts.
    backup["latency_ms"] = round(backup["latency_ms"] + result.get("latency_ms", 0), 1)
    backup["note"] = f"Prism failed ({result['problem']}), used rules"
    backup["rate_limit_retries"] = result.get("rate_limit_retries", 0)
    return backup


# ---------------------------------------------------------------------------
# The routing policy: pick a tier from the classification, and explain why in plain English.
# ---------------------------------------------------------------------------
def choose_tier(cls):
    if cls["language"] in ("indian_language", "code_mixed"):
        tier = config.INDIC_TIER
    else:
        tier = config.ENGLISH_ROUTES[cls["difficulty"]]
    moved_up = False
    if cls["difficulty_prob"] < config.CONFIDENCE_THRESHOLD and tier != "tier3":
        tier = TIERS[TIERS.index(tier) + 1]
        moved_up = True
    label = LANGUAGE_LABELS.get(cls["language"], cls["language"])
    if cls["classifier"] == "rules":
        reason = f"{label}, {cls['difficulty']} (rules) -> {config.MODELS[tier]['label']}"
        if cls.get("note", "").startswith("Prism failed"):
            reason += f" [{cls['note']}]"
    else:
        reason = (f"{label}, {cls['difficulty']}, confidence {cls['difficulty_prob']:.2f} "
                  f"-> {config.MODELS[tier]['label']}")
    if moved_up:
        reason += f" (moved up one tier: confidence below {config.CONFIDENCE_THRESHOLD:.2f})"
    return tier, reason


# ---------------------------------------------------------------------------
# Answer a prompt starting at one tier, with one fallback.
# - Rate limited even after the waits: the row is marked failed (never a fallback).
# - Timeout, error, refusal or empty answer: retry once on the next tier up.
# - Tier 3 has nothing above it, so it retries once on Tier 3 (without reusing the saved answer).
# ---------------------------------------------------------------------------
def problem_with(result):
    if result["status"] != "ok":
        return result["status"]  # timeout, error or rate_limited
    if not result["answer"]:
        return "empty"
    if is_refusal(result["answer"]):
        return "refusal"
    return None


def answer_with_fallback(start_tier, text, use_cache=True, first_timeout=None):
    messages = [{"role": "user", "content": text}]
    # Tier 3 has no tier above it, so if its provider is merely busy (503) we wait and retry it.
    first = call_model(start_tier, messages, timeout=first_timeout, read_cache=use_cache,
                       write_cache=use_cache, cache_text=text, retry_when_busy=(start_tier == "tier3"))
    problem = problem_with(first)
    if problem is None:
        return {"final": first, "final_tier": start_tier, "first": None, "fallback_reason": "",
                "status": "ok", "fail_reason": ""}
    if problem == "rate_limited":
        return {"final": first, "final_tier": start_tier, "first": None, "fallback_reason": "",
                "status": "failed", "fail_reason": "rate_limited"}

    next_tier = TIERS[min(TIERS.index(start_tier) + 1, len(TIERS) - 1)]
    same_tier = next_tier == start_tier
    print(f"      {problem} on {config.MODELS[start_tier]['label']}; "
          f"{'retrying once on the same tier' if same_tier else 'falling back to ' + config.MODELS[next_tier]['label']}")
    second = call_model(next_tier, messages, read_cache=use_cache and not same_tier,
                        write_cache=use_cache, cache_text=text, retry_when_busy=(next_tier == "tier3"))
    problem2 = problem_with(second)
    return {"final": second, "final_tier": next_tier, "first": first,
            "fallback_reason": problem + (" (retried on Tier 3)" if same_tier else ""),
            "status": "ok" if problem2 is None else "failed", "fail_reason": problem2 or ""}


# ---------------------------------------------------------------------------
# Results file: one row per prompt per configuration.
# ---------------------------------------------------------------------------
RESULT_COLUMNS = [
    "prompt_id", "set", "category", "language", "config", "route_reason", "routed_tier", "classifier",
    "pred_language", "pred_difficulty", "difficulty_prob", "model", "final_tier", "status", "fail_reason",
    "fallback", "fallback_reason", "first_attempt_model", "input_tokens", "output_tokens", "reasoning_tokens",
    "model_cost_inr", "first_attempt_cost_inr", "router_tokens", "router_cost_inr", "cost_inr",
    "model_latency_ms", "first_attempt_latency_ms", "router_latency_ms", "latency_ms",
    "rate_limit_retries", "from_cache", "finish_reason", "error", "answer", "expected_answer",
    "exact_match", "quality_score", "judge_reason", "judge_input_tokens", "judge_output_tokens", "judge_cost_inr",
]


def load_results(path=None):
    path = path or config.RESULTS_FILE
    if not os.path.exists(path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[RESULT_COLUMNS]


def save_results(df, path=None):
    # utf-8-sig lets Excel show Hindi and Tamil correctly.
    df.to_csv(path or config.RESULTS_FILE, index=False, encoding="utf-8-sig")


def merge_rows(new_rows, path=None):
    """Replace only the rows for this configuration and these prompts; keep everything else."""
    if not new_rows:
        return
    new = pd.DataFrame(new_rows).reindex(columns=RESULT_COLUMNS).fillna("").astype(str)
    old = load_results(path)
    # If a prompt got exactly the same answer as last time, keep its earlier judge score.
    judge_cols = ["quality_score", "judge_reason", "judge_input_tokens", "judge_output_tokens", "judge_cost_inr"]
    previous = {(r["config"], r["prompt_id"]): r for _, r in old.iterrows() if r["quality_score"] != ""}
    for idx, r in new.iterrows():
        before = previous.get((r["config"], r["prompt_id"]))
        if before is not None and before["answer"] == r["answer"] and r["status"] == "ok":
            for col in judge_cols:
                new.at[idx, col] = before[col]
    keys = set(zip(new["config"], new["prompt_id"]))
    keep = old[[(c, p) not in keys for c, p in zip(old["config"], old["prompt_id"])]]
    combined = pd.concat([keep, new], ignore_index=True)
    combined = combined.sort_values(["config", "set", "prompt_id"], kind="stable")
    save_results(combined, path)


def load_prompts(prompt_set):
    if not os.path.exists(config.PROMPTS_FILE):
        raise StopRun(f"{config.PROMPTS_FILE} not found. It should be in the same folder as router.py.")
    df = pd.read_csv(config.PROMPTS_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"id", "set", "category", "language", "prompt", "expected_answer"}
    missing = needed - set(df.columns)
    if missing:
        raise StopRun(f"{config.PROMPTS_FILE} is missing these columns: {', '.join(sorted(missing))}")
    df = df[df["set"].str.strip().str.lower() == prompt_set]
    if df.empty:
        raise StopRun(f"No rows with set = '{prompt_set}' in {config.PROMPTS_FILE}.")
    return df


def r2(x):
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# Run one configuration over a prompt set and build one results row per prompt.
# ---------------------------------------------------------------------------
def process_prompt(cfg, p, use_cache=True, first_timeout=None, verbose=True):
    text = p["prompt"]
    if cfg == "router":
        cls = classify(text, read_cache=use_cache, write_cache=use_cache)
        tier, reason = choose_tier(cls)
        router = {"tokens": cls["tokens"], "cost": cls["cost_inr"], "latency": cls["latency_ms"]}
    else:
        cls = None
        tier = "tier1" if cfg == "always_cheapest" else "tier3"
        reason = f"baseline: every prompt -> {config.MODELS[tier]['label']}"
        router = {"tokens": 0, "cost": 0.0, "latency": 0.0}
    if verbose:
        print(f"      {reason}")

    out = answer_with_fallback(tier, text, use_cache=use_cache, first_timeout=first_timeout)
    final, first = out["final"], out["first"]
    first_cost = first["cost_inr"] if first else 0.0
    first_latency = first["latency_ms"] if first else 0.0
    total_cost = final["cost_inr"] + first_cost + router["cost"]
    total_latency = final["latency_ms"] + first_latency + router["latency"]
    expected = p.get("expected_answer", "")
    exact = ""
    if str(expected).strip() and out["status"] == "ok":
        exact = "yes" if normalise(expected) in normalise(final["answer"]) else "no"

    return {
        "prompt_id": p["id"], "set": p["set"], "category": p["category"], "language": p["language"],
        "config": cfg, "route_reason": reason, "routed_tier": tier,
        "classifier": cls["classifier"] if cls else "",
        "pred_language": cls["language"] if cls else "", "pred_difficulty": cls["difficulty"] if cls else "",
        "difficulty_prob": r2(cls["difficulty_prob"]) if cls else "",
        "model": config.MODELS[out["final_tier"]]["model"], "final_tier": out["final_tier"],
        "status": out["status"], "fail_reason": out["fail_reason"],
        "fallback": "yes" if first else "no", "fallback_reason": out["fallback_reason"],
        "first_attempt_model": config.MODELS[tier]["model"] if first else "",
        "input_tokens": final["input_tokens"], "output_tokens": final["output_tokens"],
        "reasoning_tokens": final["reasoning_tokens"],
        "model_cost_inr": r2(final["cost_inr"]), "first_attempt_cost_inr": r2(first_cost),
        "router_tokens": router["tokens"], "router_cost_inr": r2(router["cost"]), "cost_inr": r2(total_cost),
        "model_latency_ms": final["latency_ms"], "first_attempt_latency_ms": round(first_latency, 1),
        "router_latency_ms": router["latency"], "latency_ms": round(total_latency, 1),
        "rate_limit_retries": final["rate_limit_retries"] + (first["rate_limit_retries"] if first else 0)
                              + (cls.get("rate_limit_retries", 0) if cls else 0),
        "from_cache": "yes" if final.get("from_cache") else "no",
        "finish_reason": final.get("finish_reason", ""),
        "error": (first["error"] + " | " if first and first.get("error") else "") + final.get("error", ""),
        "answer": final["answer"], "expected_answer": expected, "exact_match": exact,
        "quality_score": "", "judge_reason": "", "judge_input_tokens": "", "judge_output_tokens": "",
        "judge_cost_inr": "",
    }


def run_config(cfg, prompt_set, judge_after=True, limit=None):
    prompts = load_prompts(prompt_set)
    if limit:
        prompts = prompts.head(limit)
    total = len(prompts)
    print(f"\nRunning '{cfg}' on {total} {prompt_set} prompts...\n")
    rows = []
    try:
        for i, (_, p) in enumerate(prompts.iterrows(), 1):
            print(f"[{i}/{total}] {p['id']}  ({p['category']}, {p['language']})")
            row = process_prompt(cfg, p)
            rows.append(row)
            print(f"      -> {row['status']}, {row['model']}, Rs {float(row['cost_inr']):.4f}, "
                  f"{float(row['latency_ms']):.0f} ms{', from cache' if row['from_cache'] == 'yes' else ''}"
                  f"{', fallback: ' + row['fallback_reason'] if row['fallback'] == 'yes' else ''}")
    except StopRun as e:
        print(f"\nSTOPPED: {e}")
    except KeyboardInterrupt:
        print("\nStopped by you (Ctrl+C). Saving what is done so far.")
    finally:
        merge_rows(rows)
        print(f"\nSaved {len(rows)} rows to {config.RESULTS_FILE}.")
    failed = [r["prompt_id"] for r in rows if r["status"] == "failed"]
    if failed:
        print(f"Failed rows: {', '.join(failed)}. Rerun the same command later; only these will call the API again.")
    if len(rows) < total:
        print(f"{total - len(rows)} prompts were not run. Rerun the same command to finish; saved answers are reused.")
    print_spend()
    if judge_after and rows:
        judge_results(cfg, prompt_set)


# ---------------------------------------------------------------------------
# The judge: a different model family (Qwen) scores each answer from 1 to 5.
# It never sees which model wrote the answer or which configuration it came from.
# Judge cost is tracked separately: it is a measurement cost, not a serving cost.
# ---------------------------------------------------------------------------
JUDGE_INSTRUCTIONS = """You are grading an AI assistant's answer to a task from an Indian enterprise
(banking, insurance, customer support, documents or translation).

Score the answer from 1 to 5 on three things together:
- correctness: are the facts, numbers, labels and reasoning right?
- completeness: does it do everything the task asked?
- language: is it in the language and script the user wrote in or asked for?

5 = correct, complete, right language and script
4 = correct with small gaps
3 = partly correct or noticeably incomplete
2 = mostly wrong, or in the wrong language or script
1 = wrong, empty, or a refusal

If a reference answer is given, use it to check correctness.
Reply with JSON only, nothing else: {"score": <1-5>, "reason": "<one short line>"}"""


def parse_judge_reply(text):
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        score = int(round(float(data["score"])))
        if 1 <= score <= 5:
            return score, str(data.get("reason", "")).strip()
    except Exception:
        return None
    return None


def judge_one(prompt, answer, expected):
    """Returns (score or None, reason, input_tokens, output_tokens, cost, stop_reason)."""
    key = cache_key(config.MODELS["judge"]["model"], f"{prompt}||{answer}")
    hit = load_cache(config.JUDGE_CACHE_FILE).get(key)
    if hit:
        r = hit["result"]
        return r["score"], r["reason"], r["input_tokens"], r["output_tokens"], r["cost_inr"], ""

    task = f"TASK:\n{prompt}\n\nREFERENCE ANSWER: {expected if str(expected).strip() else '(none)'}\n\nANSWER TO GRADE:\n{answer}"
    messages = [{"role": "system", "content": JUDGE_INSTRUCTIONS}, {"role": "user", "content": task}]
    tokens_in = tokens_out = 0
    cost = 0.0
    for attempt in (1, 2):  # if the reply is not valid JSON, ask once more
        res = call_model("judge", messages, read_cache=False, write_cache=False,
                         rate_limit_wait=config.JUDGE_RATE_LIMIT_WAIT_SECONDS, retry_when_busy=True)
        if res["status"] == "rate_limited":
            return None, "", tokens_in, tokens_out, cost, "rate_limited"
        tokens_in += res["input_tokens"]
        tokens_out += res["output_tokens"]
        cost += res["cost_inr"]
        time.sleep(max(0, config.JUDGE_PAUSE_SECONDS - provider_pause(config.MODELS["judge"]["provider"])))
        parsed = parse_judge_reply(res["answer"]) if res["status"] == "ok" else None
        if parsed:
            score, reason = parsed
            save_to_cache(config.JUDGE_CACHE_FILE, key, {"score": score, "reason": reason, "input_tokens": tokens_in,
                                                         "output_tokens": tokens_out, "cost_inr": cost})
            return score, reason, tokens_in, tokens_out, cost, ""
    reason = "judge parse error" if res["status"] == "ok" else f"judge {res['status']}: {short(res['error'], 120)}"
    return None, reason, tokens_in, tokens_out, cost, ""


def judge_results(cfg, prompt_set):
    df = load_results()
    todo = df[(df["config"] == cfg) & (df["set"] == prompt_set) & (df["status"] == "ok")
              & (df["quality_score"] == "")]
    if todo.empty:
        print(f"\nNothing to judge for '{cfg}' on the {prompt_set} set.")
        return
    print(f"\nJudging {len(todo)} answers with {config.MODELS['judge']['label']}...")
    done = 0
    try:
        for n, (idx, row) in enumerate(todo.iterrows(), 1):
            prompt = load_prompt_text(row["prompt_id"])
            score, reason, t_in, t_out, cost, stop = judge_one(prompt, row["answer"], row["expected_answer"])
            if stop == "rate_limited":
                print("\nThe judge is still rate-limited after the waits; its free daily limit is probably used up.")
                break
            df.at[idx, "quality_score"] = "" if score is None else str(score)
            df.at[idx, "judge_reason"] = reason
            df.at[idx, "judge_input_tokens"] = str(t_in)
            df.at[idx, "judge_output_tokens"] = str(t_out)
            df.at[idx, "judge_cost_inr"] = str(r2(cost))
            done += 1
            print(f"  [{n}/{len(todo)}] {row['prompt_id']}: {score if score is not None else '-'}  {short(reason, 90)}")
    except StopRun as e:
        print(f"\nSTOPPED: {e}")
    except KeyboardInterrupt:
        print("\nStopped by you (Ctrl+C).")
    finally:
        save_results(df)
    left = len(todo) - done
    print(f"\nJudged {done} answer{'' if done == 1 else 's'}." + (f" {left} still need a score: run "
                                          f"'python router.py --judge --config {cfg} --set {prompt_set}' later "
                                          f"(for example tomorrow)." if left else ""))


_prompt_text_lookup = None


def load_prompt_text(prompt_id):
    global _prompt_text_lookup
    if _prompt_text_lookup is None:
        df = pd.read_csv(config.PROMPTS_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]
        _prompt_text_lookup = dict(zip(df["id"], df["prompt"]))
    return _prompt_text_lookup.get(prompt_id, "")


# ---------------------------------------------------------------------------
# The summary table: one row per configuration, test set only.
# ---------------------------------------------------------------------------
def num(series):
    return pd.to_numeric(series, errors="coerce")


def tier_share(series):
    counts = series.value_counts(normalize=True)
    return " / ".join(f"{t.replace('tier', 'T')} {counts.get(t, 0) * 100:.0f}%" for t in TIERS)


def summarize():
    df = load_results()
    df = df[df["set"] == "test"]
    if df.empty:
        print("No test-set results yet. Run the three configurations on --set test first.")
        return
    rows = []
    for cfg in CONFIGS:
        d = df[df["config"] == cfg]
        if d.empty:
            continue
        ok = d[d["status"] == "ok"]
        q = num(ok["quality_score"]).dropna()
        with_expected = ok[ok["expected_answer"].str.strip() != ""]
        row = {
            "config": cfg,
            "prompts": len(d),
            "total_cost_inr": round(num(d["cost_inr"]).sum(), 4),
            "avg_cost_per_prompt_inr": round(num(d["cost_inr"]).mean(), 5),
            "p95_latency_ms": round(num(ok["latency_ms"]).quantile(0.95), 0) if len(ok) else "",
            "avg_quality": round(q.mean(), 2) if len(q) else "",
            "share_scoring_4_or_5": f"{(q >= 4).mean() * 100:.0f}%" if len(q) else "",
            "judged": f"{len(q)} of {len(ok)}",
            "exact_match_rate": (f"{(with_expected['exact_match'] == 'yes').mean() * 100:.0f}%"
                                 if len(with_expected) else ""),
            "fallbacks": int((d["fallback"] == "yes").sum()),
            "failed": int((d["status"] == "failed").sum()),
            "rate_limit_retries": int(num(d["rate_limit_retries"]).fillna(0).sum()),
            "judge_cost_inr": round(num(d["judge_cost_inr"]).fillna(0).sum(), 4),
            "router_cost_inr": round(num(d["router_cost_inr"]).fillna(0).sum(), 4),
            "routed_share_by_prompts": tier_share(d["routed_tier"]) if cfg == "router" else "",
        }
        if cfg == "router":
            # Share of tokens answered by each tier: the traffic mix to use in Part 2.
            tokens = num(d["input_tokens"]).fillna(0) + num(d["output_tokens"]).fillna(0)
            by_tier = tokens.groupby(d["final_tier"]).sum()
            total = by_tier.sum() or 1
            row["token_share_by_tier"] = " / ".join(f"{t.replace('tier', 'T')} {by_tier.get(t, 0) / total * 100:.0f}%"
                                                    for t in TIERS)
        else:
            row["token_share_by_tier"] = ""
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(config.SUMMARY_FILE, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nSUMMARY (test set)\n")
    print(summary.set_index("config").T.to_string())
    print(f"\nSaved to {config.SUMMARY_FILE}. Note: average quality covers judged rows only; failed rows are counted separately.")

    # The "hardest tenth": test prompts where the cheapest model scored lowest.
    cheap = df[df["config"] == "always_cheapest"].copy()
    cheap["q"] = num(cheap["quality_score"])
    cheap = cheap.dropna(subset=["q"])
    if cheap.empty:
        return
    n = max(1, math.ceil(df["prompt_id"].nunique() * config.HARDEST_SHARE))
    hardest = cheap.sort_values(["q", "prompt_id"]).head(n)
    lines = []
    for _, h in hardest.iterrows():
        line = {"prompt_id": h["prompt_id"], "category": h["category"], "cheapest_score": int(h["q"])}
        for cfg in ("always_frontier", "router"):
            m = df[(df["config"] == cfg) & (df["prompt_id"] == h["prompt_id"])]
            if not m.empty:
                line[f"{cfg}_score"] = m.iloc[0]["quality_score"] or "-"
                line[f"{cfg}_cost_inr"] = m.iloc[0]["cost_inr"]
                if cfg == "router":
                    line["router_model"] = m.iloc[0]["model"]
        lines.append(line)
    print(f"\nHARDEST TENTH: the {n} test prompts where always_cheapest scored lowest\n")
    print(pd.DataFrame(lines).to_string(index=False))


# ---------------------------------------------------------------------------
# Check that every model answers.
# ---------------------------------------------------------------------------
def check_models():
    questions = [("English", "Reply in one sentence: what is a fixed deposit?"),
                 ("Hindi", "एक वाक्य में बताइए: फिक्स्ड डिपॉज़िट क्या है?")]
    for model_key in ("tier1", "tier2", "tier3", "judge"):
        m = config.MODELS[model_key]
        print(f"\n=== {m['label']}  ({m['model']} on {m['provider']}) ===")
        for lang, question in questions:
            try:
                res = call_model(model_key, [{"role": "user", "content": question}], read_cache=False,
                                 write_cache=False, retry_when_busy=(model_key in ("judge", "tier3")))
            except StopRun as e:
                print(f"  {lang}: STOPPED - {e}")
                break
            if res["status"] != "ok":
                print(f"  {lang}: {res['status'].upper()} - {res['error']}")
                continue
            print(f"  {lang}: {short(res['answer'], 200)}")
            print(f"      input {res['input_tokens']} tokens, output {res['output_tokens']} tokens "
                  f"(of which thinking {res['reasoning_tokens']}), cost Rs {res['cost_inr']:.5f}, "
                  f"latency {res['latency_ms']:.0f} ms, finish: {res['finish_reason']}")
    print_spend()


# ---------------------------------------------------------------------------
# Show how five different kinds of prompt are routed (and compare with Prism, if it is on).
# ---------------------------------------------------------------------------
SAMPLE_PROMPTS = [
    "Classify this complaint as billing, fraud or service: 'I was charged twice for my credit card annual fee.'",
    "Bhai mera UPI payment fail ho gaya but paisa account se kat gaya, kya karun? Jaldi batao.",
    "मेरे होम लोन की EMI कब कटेगी? कृपया तारीख बताइए।",
    "Compare a term insurance plan with a ₹1 crore cover at ₹12,000 a year against an endowment plan with "
    "₹25 lakh cover at ₹60,000 a year for a 30-year-old with two children, and recommend one with reasons.",
    "Translate this SMS into Marathi: 'Your loan EMI of ₹8,500 is due on 5 December. Please keep your account funded.'",
]


def classify_test():
    for text in SAMPLE_PROMPTS:
        print(f"\nPROMPT: {short(text, 110)}")
        if config.CLASSIFIER == "rules":
            rules = classify_rules(text)
            tier, reason = choose_tier(rules)
            print(f"  Rules: language={rules['language']}, difficulty={rules['difficulty']}")
            print(f"         route: {reason}")
            continue
        try:
            prism = classify_prism(text, read_cache=False, write_cache=False)
        except StopRun as e:
            print(f"  Prism: STOPPED - {e}")
            return
        if prism["ok"]:
            tier, reason = choose_tier(prism)
            print(f"  Prism: language={prism['language']} ({prism['language_prob']:.2f}), "
                  f"difficulty={prism['difficulty']} ({prism['difficulty_prob']:.2f}), "
                  f"{prism['tokens']} tokens, Rs {prism['cost_inr']:.5f}, {prism['latency_ms']:.0f} ms")
            print(f"         route: {reason}")
        else:
            print(f"  Prism: FAILED - {prism['problem']}")
        rules = classify_rules(text)
        tier, reason = choose_tier(rules)
        print(f"  Rules: language={rules['language']}, difficulty={rules['difficulty']}  route: {reason}")
    print_spend()


# ---------------------------------------------------------------------------
# Force a fallback. The first attempt gets a 0.01-second timeout, which always fails,
# so each prompt must fall back. Uses no cache and writes to its own file, so results.csv is untouched.
# ---------------------------------------------------------------------------
def fallback_test(n=3):
    prompts = load_prompts("practice").head(n)
    rows = []
    print(f"\nForcing a fallback on {len(prompts)} practice prompts (first attempt gets a 0.01 s timeout)...\n")
    try:
        for _, p in prompts.iterrows():
            print(f"{p['id']}:")
            row = process_prompt("router", p, use_cache=False, first_timeout=0.01)
            rows.append(row)
            print(f"{row['prompt_id']}: first attempt {row['first_attempt_model'] or '-'} -> "
                  f"fallback={row['fallback']} ({row['fallback_reason']}), answered by {row['model']}, "
                  f"status {row['status']}")
    except StopRun as e:
        print(f"\nSTOPPED: {e}")
    finally:
        if rows:
            pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS).to_csv(config.FALLBACK_TEST_FILE, index=False,
                                                                       encoding="utf-8-sig")
            print(f"\nSaved to {config.FALLBACK_TEST_FILE} (results.csv was not changed).")
    print_spend()


# ---------------------------------------------------------------------------
# Probe: try many free models once with the same Hindi question, to see which ones actually
# answer, how fast, and how good their Hindi is. Used to choose the lineup; not part of the test.
# ---------------------------------------------------------------------------
PROBE_CANDIDATES = [
    ("groq", "openai/gpt-oss-120b"),
    ("groq", "qwen/qwen3.8-27b"),
    ("nvidia", "moonshotai/kimi-k3.1-pro"),
    ("nvidia", "moonshotai/kimi-k3"),
    ("nvidia", "google/gemma-4-31b-it"),
    ("nvidia", "deepseek-ai/deepseek-v4.1-flash"),
    ("nvidia", "mistralai/mistral-large-4-1t-a52b"),
    ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
    ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
    ("nvidia", "minimaxai/minimax-m3.1-flash"),
    ("nvidia", "stepfun-ai/step-5-preview"),
    ("nvidia", "z-ai/glm-5.3-flash"),
    ("gemini", "gemini-3.8-flash"),
    ("gemini", "gemini-3.6-flash"),
    ("gemini", "gemma-4-31b-it"),
]
PROBE_QUESTION = "हिंदी में दो वाक्यों में बताइए: फिक्स्ड डिपॉज़िट क्या है और इसमें क्या जोखिम है?"
PROBE_TIMEOUT_SECONDS = 40


def probe_models():
    print(f"\nTrying {len(PROBE_CANDIDATES)} free models with one Hindi question "
          f"(up to {PROBE_TIMEOUT_SECONDS}s each, so this can take several minutes)...\n")
    summary = []
    for provider, model in PROBE_CANDIDATES:
        print(f"--- {model} ({provider})")
        try:
            client = get_client(provider)
        except StopRun as e:
            print(f"    skipped: {e}")
            summary.append((provider, model, "skipped (no key)", ""))
            continue
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": PROBE_QUESTION}],
                max_tokens=1500, timeout=PROBE_TIMEOUT_SECONDS)
            latency_ms = (time.perf_counter() - start) * 1000
            choice = response.choices[0] if response.choices else None
            answer = clean_answer(choice.message.content if choice and choice.message else "")
            usage = response.usage
            out_tokens = (usage.completion_tokens or 0) if usage else 0
            status = "OK" if answer else "EMPTY ANSWER"
            print(f"    {status}, {latency_ms:.0f} ms, {out_tokens} output tokens")
            if answer:
                print(f"    {short(answer, 220)}")
            summary.append((provider, model, status, f"{latency_ms:.0f} ms"))
        except openai.APITimeoutError:
            print(f"    TIMEOUT after {PROBE_TIMEOUT_SECONDS}s")
            summary.append((provider, model, "timeout", ""))
        except openai.APIStatusError as e:
            print(f"    ERROR {e.status_code}: {short(e, 150)}")
            summary.append((provider, model, f"error {e.status_code}", ""))
        except openai.APIConnectionError as e:
            print(f"    CONNECTION ERROR: {short(e, 150)}")
            summary.append((provider, model, "connection error", ""))
        time.sleep(provider_pause(provider))
    print("\nSUMMARY")
    for provider, model, status, latency in summary:
        print(f"  {status:18s} {latency:>9s}  {model} ({provider})")


# ---------------------------------------------------------------------------
# List the models a provider makes available to your key (to pick exact model names).
# ---------------------------------------------------------------------------
def list_models(provider, word=""):
    client = get_client(provider)
    try:
        ids = sorted(m.id for m in client.models.list())
    except openai.APIStatusError as e:
        print(f"Could not list models on {provider}: {e.status_code} {short(e, 200)}")
        return
    except openai.APIConnectionError as e:
        print(f"Could not reach {provider}: {short(e, 200)}")
        return
    if word:
        ids = [i for i in ids if word.lower() in i.lower()]
    print(f"\n{len(ids)} models on {provider}" + (f" containing '{word}'" if word else "") + ":\n")
    for model_id in ids:
        print(f"  {model_id}")


# ---------------------------------------------------------------------------
# Spend report from the call log.
# ---------------------------------------------------------------------------
def spend_report():
    if not os.path.exists(config.CALL_LOG_FILE):
        print("No API calls logged yet.")
        return
    log = pd.read_csv(config.CALL_LOG_FILE, encoding="utf-8")
    table = log.groupby(["provider", "model"]).agg(calls=("cost_inr", "size"), input_tokens=("input_tokens", "sum"),
                                                   output_tokens=("output_tokens", "sum"), cost_inr=("cost_inr", "sum"))
    print("\nLive API calls so far (list-price cost; free-tier calls cost nothing):\n")
    print(table.round(4).to_string())
    print_spend()


# ---------------------------------------------------------------------------
# Command-line options.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model router for the Sarvam GTM assignment")
    parser.add_argument("--check", action="store_true", help="test every model once")
    parser.add_argument("--classify-test", action="store_true", help="show how 5 sample prompts would be routed")
    parser.add_argument("--config", choices=CONFIGS, help="which configuration to run or judge")
    parser.add_argument("--set", choices=["practice", "test"], help="which prompts to use")
    parser.add_argument("--no-judge", action="store_true", help="run without judging afterwards")
    parser.add_argument("--judge", action="store_true", help="only judge rows that have no score yet")
    parser.add_argument("--limit", type=int, help="only use the first N prompts")
    parser.add_argument("--summary", action="store_true", help="write and print summary.csv")
    parser.add_argument("--fallback-test", action="store_true", help="force a fallback on 3 practice prompts")
    parser.add_argument("--spend", action="store_true", help="show the cost of all live API calls so far")
    parser.add_argument("--list-models", nargs="?", const="", metavar="WORD",
                        help="list model names on a provider, optionally only those containing WORD")
    parser.add_argument("--provider", default="nvidia", choices=sorted(config.PROVIDERS),
                        help="provider for --list-models (default: nvidia)")
    parser.add_argument("--probe", action="store_true", help="try many free models once to see which work")
    args = parser.parse_args()

    try:
        if args.check:
            check_models()
        elif args.classify_test:
            classify_test()
        elif args.judge:
            if not (args.config and args.set):
                parser.error("--judge needs --config and --set")
            judge_results(args.config, args.set)
        elif args.config:
            if not args.set:
                parser.error("--config needs --set practice or --set test")
            run_config(args.config, args.set, judge_after=not args.no_judge, limit=args.limit)
        elif args.summary:
            summarize()
        elif args.fallback_test:
            fallback_test()
        elif args.spend:
            spend_report()
        elif args.list_models is not None:
            list_models(args.provider, args.list_models)
        elif args.probe:
            probe_models()
        else:
            parser.print_help()
    except StopRun as e:
        print(f"\nSTOPPED: {e}")


if __name__ == "__main__":
    main()
