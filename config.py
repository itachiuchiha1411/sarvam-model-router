# config.py
# Every setting and number the router uses lives in this one file, so each
# price, threshold and timeout can be checked and defended in one place.
# Prices were checked on PRICES_CHECKED_ON (below) on the pages named in the comments.
#
# Budget: Rs 0. Every model runs on a free tier (Groq, NVIDIA). Costs in the
# results are what each call WOULD cost at the public list price, so the comparison is real.

from dotenv import load_dotenv

# Read the API keys from the .env file. Keys are never written in this file.
load_dotenv()

# ---------------------------------------------------------------------------
# Where each provider's API lives, and which .env key it uses.
# All of them speak the same "OpenAI-compatible" format, so one library talks to all of them.
# ---------------------------------------------------------------------------
PROVIDERS = {
    # Groq's OpenAI-compatible API (base URL) (console.groq.com/docs)
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    # NVIDIA's free API for prototyping (build.nvidia.com): no daily cap, about 40 requests a minute
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY"},
    # Google's OpenAI-compatible API (base URL). Not used at the moment: in mid-November 2026 its free tier
    # was returning frequent 500/503 errors and timeouts for Gemma 4 and Gemini 3.8 Flash.
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY"},
    # Not used at the moment. Kept so Sarvam models can be switched back in (see the end of MODELS).
    "sarvam": {"base_url": "https://api.sarvam.ai/v2", "key_env": "SARVAM_API_KEY"},
}

# Sarvam Prism's "Decisions API" address. Only used if CLASSIFIER = "prism".
PRISM_URL = "https://api.sarvam.ai/v2/systemone"

# ---------------------------------------------------------------------------
# The models. Prices are per 1 million tokens.
# currency "INR" = rupees as listed; "USD" = dollars, converted with USD_TO_INR below.
# reasoning_effort None = use each model's own default thinking level.
# ---------------------------------------------------------------------------
MODELS = {
    # Tier 1, cheapest and fastest. Free on Groq's free tier; priced at Groq's paid list price.
    # Source: groq.com/pricing
    "tier1": {"label": "gpt-oss-20b (Groq)", "provider": "groq", "model": "openai/gpt-oss-20b",
              "currency": "USD", "price_in": 0.075, "price_out": 0.30, "reasoning_effort": None},

    # Tier 2, the Indian-language and middle tier: gpt-oss-120b, the larger sibling of Tier 1.
    # Free on Groq's free tier; priced at Groq's paid list price. In our probe it answered a Hindi
    # question well in under a second. Source: groq.com/pricing
    "tier2": {"label": "gpt-oss-120b (Groq)", "provider": "groq", "model": "openai/gpt-oss-120b",
              "currency": "USD", "price_in": 0.15, "price_out": 0.60, "reasoning_effort": None},

    # Tier 3, frontier: NVIDIA's Nemotron 3 Ultra (550B parameters), run free on NVIDIA's API.
    # The only large model that answered reliably in our probe. Priced at the median list price across
    # the providers that sell it: $0.60 in, $2.70 out (artificialanalysis.ai, checked 14 Nov 2026).
    "tier3": {"label": "Nemotron 3 Ultra (NVIDIA)", "provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b",
              "currency": "USD", "price_in": 0.60, "price_out": 2.70, "reasoning_effort": None},

    # Tried and dropped on 14 Nov 2026 (python router.py --probe), also listed in the README:
    #   Sarvam 105B and Prism: no Sarvam signup credit arrived (error 402).
    #   Gemini 3.8 Flash and Gemma 4 31B (Google): repeated 503/500 errors and timeouts on the free tier.
    #   Kimi K3.1 Pro, Kimi K2.6, Nemotron 3 Super (NVIDIA): listed, but "not found" when called.
    #   Gemma 4, DeepSeek V4.1 Flash, Mistral Large 4, MiniMax M3.1, Step 5, GLM-5.3 Flash (NVIDIA):
    #   no reply within 40 seconds. Gemini 3.6 Flash worked but took about 15 seconds.

    # The quality judge: a different model family (Alibaba's Qwen) from all three tiers, so no
    # model grades its own answers. Free on Groq's free tier; priced at Groq's list price.
    # Source: console.groq.com/docs/models (Preview models)
    "judge": {"label": "Qwen 3.8 27B (Groq, judge)", "provider": "groq", "model": "qwen/qwen3.8-27b",
              "currency": "USD", "price_in": 0.80, "price_out": 4.00, "reasoning_effort": None},

    # Sarvam Prism, the classifier used if CLASSIFIER = "prism" (needs Sarvam credit).
    # Source: docs.sarvam.ai/api/pricing
    "router": {"label": "Sarvam Prism", "provider": "sarvam", "model": "sarvam-prism",
               "currency": "INR", "price_in": 4.0, "price_out": 0.0},

    # To switch back to Sarvam models later (needs Sarvam credit), replace the tier2 entry above with:
    # "tier2": {"label": "Sarvam 105B", "provider": "sarvam", "model": "sarvam-105b",
    #           "currency": "INR", "price_in": 15.0, "price_out": 60.0, "reasoning_effort": None},
    # and set CLASSIFIER = "prism" below.
}

# Dollar-to-rupee rate: RBI reference rate for 13 Nov 2026 (published by FBIL).
USD_TO_INR = 94.1997
PRICES_CHECKED_ON = "2026-11-14"

# Sarvam's free signup credit; only shown if a Sarvam model is in use.
SARVAM_FREE_CREDIT_INR = 100.0

# ---------------------------------------------------------------------------
# The routing policy, in one place.
# ---------------------------------------------------------------------------
# Prompts in an Indian language, or code-mixed (e.g. Hinglish), or asking for an answer in an
# Indian language, go to this tier.
INDIC_TIER = "tier2"
# English prompts go to a tier based on how hard the classifier thinks they are.
ENGLISH_ROUTES = {"simple": "tier1", "moderate": "tier2", "complex": "tier3"}
# Which classifier to use:
#   "rules" = simple rules (Indian-script letters, Hinglish words, keywords, length). Free, instant.
#   "prism" = Sarvam Prism, which also gives a confidence score (needs Sarvam credit).
CLASSIFIER = "rules"
# Prism only: if it is less sure than this about difficulty, move the prompt up one tier.
# (The rules give no confidence score, so this has no effect while CLASSIFIER = "rules".)
CONFIDENCE_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Limits and safety settings
# ---------------------------------------------------------------------------
# Give up on a model reply after this many seconds, and fall back to the next tier.
# 90 rather than 30: thinking models on shared free endpoints can take over a minute on hard prompts.
# A real-time voice agent would need a far lower limit (see question 3).
TIMEOUT_SECONDS = 90
# Maximum length of each reply. High on purpose: thinking models can use up a small budget
# on thinking and leave the visible answer empty.
MAX_TOKENS = 4000
# Free tiers say "too many requests" (error 429) when we go too fast. Then we wait and retry
# the SAME model; this is never counted as a fallback.
RATE_LIMIT_WAIT_SECONDS = 30
RATE_LIMIT_RETRIES = 3
JUDGE_RATE_LIMIT_WAIT_SECONDS = 60
# Pause after each live API call, to stay under free-tier limits.
# Groq allows 30 requests a minute; Google's free tier allows fewer.
PAUSE_BETWEEN_CALLS_SECONDS = 2
# Groq's free tier also allows only about 8,000 tokens a minute per model, so its pause is 6 seconds.
PAUSE_BY_PROVIDER = {"groq": 6, "nvidia": 2, "gemini": 6}
# Pause between judge calls: the judge's free tier allows about 8,000 tokens a minute.
JUDGE_PAUSE_SECONDS = 12
# A reply that starts with one of these is treated as a refusal and retried one tier up.
REFUSAL_STARTS = ("i can't", "i cannot", "i can’t", "i'm unable", "i’m unable", "i am unable",
                  "i'm sorry, but i can", "i’m sorry, but i can")
# The "hardest tenth": the share of test prompts where the cheapest model scored lowest.
HARDEST_SHARE = 0.10

# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
PROMPTS_FILE = "prompts.csv"          # the test set
RESULTS_FILE = "results.csv"          # one row per prompt per configuration
SUMMARY_FILE = "summary.csv"          # one row per configuration
CACHE_FILE = "cache.jsonl"            # saved model answers, reused instead of calling twice
JUDGE_CACHE_FILE = "judge_cache.jsonl"  # saved judge scores
CALL_LOG_FILE = "api_calls_log.csv"   # every live API call, with tokens and list-price cost
FALLBACK_TEST_FILE = "fallback_test.csv"  # output of the forced-fallback test
