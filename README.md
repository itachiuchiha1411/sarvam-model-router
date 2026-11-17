# Model Router for an Indian enterprise inference platform

**Sarvam GTM & Strategy (Model API) take-home**

**Video:** [design decisions and customer pitch](https://drive.google.com/file/d/1QXsUB4DrIRGfY8nj2NZRQlkrZpaLTDy3/view?usp=drivesdk)

## Summary

I built a small router that sends each request to one of three models, from cheap to frontier, based on its language and difficulty. It logs the route, the reason, the model, the tokens, the cost, the latency, the quality score and any fallback for every request. On 40 held-out test prompts, the router cost **52% less than sending everything to the frontier model**, with **almost identical quality** (4.70 vs 4.72 out of 5). At the brief's volume of 50M input and 10M output tokens a month, the same traffic mix saves **59%: ₹2,203 instead of ₹5,369 a month**.

| Configuration | Total cost, 40 prompts | p95 latency | Median latency | Avg quality (1–5) | Scoring 4 or 5 | Exact match* | Fallbacks |
|---|---|---|---|---|---|---|---|
| Always cheapest | ₹0.86 | 2.4 s | 0.8 s | 4.58 | 88% | 95% | 1 |
| Always frontier | ₹3.54 | 32.0 s | 4.4 s | 4.72 | 95% | 100% | 0 |
| **Router** | **₹1.71** | **28.5 s** | **0.7 s** | **4.70** | **92%** | **100%** | **0** |

\*Share of the 21 test prompts that have one correct answer (a label, a number or a code) where the answer contained it.

The whole build cost **₹0**. Every model ran on a free tier. All costs in this README are what the same calls would cost at public list prices.

---

## Part 1: The router

### Stack

- Python, with the `openai`, `httpx`, `pandas` and `python-dotenv` libraries. One script, `router.py`; every number lives in `config.py`.
- Prices are per 1M tokens, checked on **14 Nov 2026**. Dollar prices are converted at **₹94.20 per US$**, the RBI reference rate for 13 Nov 2026.

| Role | Model | Where it ran (free) | List price per 1M tokens, input / output | Price source |
|---|---|---|---|---|
| Router | Keyword rules | Locally | ₹0 | – |
| Tier 1, cheapest | gpt-oss-20b | Groq free tier | $0.075 / $0.30 (₹7.06 / ₹28.26) | Groq list price |
| Tier 2, Indian-language and middle tier | gpt-oss-120b | Groq free tier | $0.15 / $0.60 (₹14.13 / ₹56.52) | Groq list price |
| Tier 3, frontier | Nemotron 3 Ultra (550B) | NVIDIA free API | $0.60 / $2.70 (₹56.52 / ₹254.34) | Median across providers (Artificial Analysis) |
| Quality judge | Qwen 3.8 27B | Groq free tier | $0.80 / $4.00 | Groq list price |

**Price sources** (checked 14 Nov 2026):

- **gpt-oss-20b, gpt-oss-120b and Qwen 3.8 27B:** [Groq pricing](https://groq.com/pricing).
- **Nemotron 3 Ultra:** [Artificial Analysis](https://artificialanalysis.ai/models/nvidia-nemotron-3-ultra-550b-a55b), median price across the providers that host it. It is an open-weight model with no official price, so each host sets its own; output prices range from about $2.20 to $3.60 per 1M tokens, and the median had moved to $0.60 / $2.40 by 16 Nov. Across that whole range, the Part 2 saving stays between 56% and 59%.
- **Exchange rate:** ₹94.20 per US$, the [RBI reference rate](https://www.msei.in/markets/Currency/Historical-Data/RBIReferenceRateArchives.aspx) for 13 Nov 2026.
- **Sarvam Prism, Sarvam 105B and GLM-5.3** (used in the Part 2 Sarvam scenario): [Sarvam API pricing](https://docs.sarvam.ai/api/pricing).

The three tiers have clearly different profiles: 20B, 120B and 550B parameters, median latency of about 0.8 s, 0.8 s and 4.4 s, and output prices that rise about 9x from bottom to top. Tiers 1 and 2 are similarly fast on Groq's hardware, but Tier 2 is a much larger model and costs twice as much. The judge is from a different model family from all three tiers, so no model grades its own answers.

**Why not Sarvam's own models?** The plan was Sarvam Prism as the router, Sarvam 105B as Tier 2 and GLM-5.3 on Sarvam's platform as Tier 3. No signup credit arrived on my account (every call returned 402), and my budget was ₹0. The router doesn't depend on any particular model: `config.py` has the lines ready to switch Sarvam 105B and Prism back in. Part 2 also prices the router on Sarvam's own rate card.

### Routing policy

Each prompt is classified on two things: language and difficulty. If the prompt is in an Indian language, is code-mixed (such as Hinglish), or asks for an answer in an Indian language (for example "Translate this into Marathi"), it goes to Tier 2. English prompts go by difficulty. Tasks with words like "compare", "calculate", "recommend" or "step by step" go to Tier 3. Summaries, translations, explanations and inputs over 80 words go to Tier 2. Everything else, meaning short English prompts (80 words or fewer) with none of these words, such as "Classify this complaint" or "Extract the EMI amount", goes to Tier 1. That was 12 of the 40 test prompts. If the chosen model times out, returns an error, refuses or returns an empty answer, the request is retried once on the next tier up and logged as a fallback. A failure on Tier 3 is retried once on Tier 3. The classifier is the part of the router that labels each prompt. Here it is a set of keyword rules that runs on my own machine, so it costs ₹0 and took 0.64 ms per request on average (4.2 ms at most). That cost and time are included in every request's totals. Keyword rules give a yes-or-no answer, not a measure of how sure they are, so there is no confidence threshold. With Sarvam Prism, which returns a probability for each label, prompts it is less than 70% sure about would move up one tier.

### Logging and failure handling

Every request writes one row to `results.csv`. The row records:
- **What it was:** prompt id, category, language, configuration.
- **Why it went where it did:** a route reason in plain English, for example "Indian language, simple (rules) -> gpt-oss-120b (Groq)", plus the predicted language and difficulty.
- **What answered:** the final model and tier, and the status.
- **Fallback details:** whether one happened, why, and which model tried first.
- **Tokens:** input, output and thinking (reasoning) tokens, as reported by each provider, never estimated.
- **Cost:** for the model, any first attempt and the router, plus the total.
- **Latency:** for the model, any first attempt and the router, plus the total.
- **Retries and reuse:** rate-limit retries, and whether the answer came from the cache.
- **The answer itself, and its quality:** the answer, the expected answer, exact match, the judge's score and reason, and the judge's tokens and cost.

Failure rules:
- **Timeout (90 s), error, refusal or empty answer:** retry once, one tier up.
- **Rate limit (429) from a free tier:** wait and retry the *same* model, up to three times. This is logged separately and never counted as a fallback.
- **A rejected key, no credit or an unknown model:** stop the run with a clear message, rather than hiding a setup error behind a fallback.
- **A thinking model that uses up its reply budget** and returns an empty answer counts as a failure and falls back. This happened once in the always-cheapest run (T37); the Tier 2 answer that replaced it was also cut off and scored 1.
- **Every successful answer is cached**, keyed by model and prompt. A rerun never pays twice, and configurations that pick the same model reuse the same answer, which keeps the comparison consistent. In the router's test run, 17 of the 40 answers were reused this way (the `from_cache` column).

`python router.py --fallback-test` forces a timeout on three prompts. All three fell back to the next tier and were answered (`fallback_test.csv`).

### Test set and tuning

The 50 prompts (`prompts.csv`) are synthetic and fictional, written with AI help and reviewed by me. There are 10 practice prompts and 40 held-out test prompts:
- 8 simple English classification and extraction tasks
- 6 call-centre transcripts of about 270 words to summarise
- 5 translations from English into Marathi, Tamil, Bengali, Hindi and Kannada
- 10 tasks in Hindi, Tamil, Bengali or Hinglish
- 5 questions about short loan, insurance and card documents
- 6 multi-step reasoning tasks

Numeric answers were calculated and spot-checked by hand. Tamil and Bengali prompts were not reviewed by a fluent reader; their quality scores rely on the AI judge alone.
I tuned the rules only on the 10 practice prompts, and made one change. "Reply" and "write" were on the list of words that send a prompt to Tier 2, to catch drafting tasks like "Write a reply to this customer". But simple classification prompts say "Reply with the category only", so they were going to Tier 2 as well. I removed both words; drafting tasks are still caught by "draft".

When I made that change, I could already see which tier each test prompt would go to. To avoid tailoring the rules to the test, I made no further changes and kept the three remaining misroutes in the results (see "What the router gets wrong" below). A real evaluation would use customer traffic labelled by someone other than the router's author; this set is not that.

### Measuring quality, and what it misses

Each answer is scored from 1 to 5 by an AI judge (Qwen 3.8 27B). The score covers correctness, completeness, and whether the answer uses the language and script the user wrote in or asked for. The judge never sees which model wrote the answer. Where a prompt has one correct answer, the script also checks whether the answer contains it (exact match). Judge cost is tracked separately, as a measurement cost rather than a serving cost.

What this measure does not capture:
- **No native-speaker review.** An AI judge grading Hindi, Tamil, Bengali, Marathi or Kannada is a weak stand-in for a fluent reader. It is sometimes simply wrong: on T15, it marked down a Marathi translation for "using Devanagari instead of the standard Marathi script", but Marathi *is* written in Devanagari.
- **Judge bias.** AI judges tend to prefer longer answers and their own style, and the judge (27B) is smaller than the frontier model it grades.
- **Small sample.** The "hardest tenth" is four prompts; that is an anecdote, not a statistic. A difference like 92% vs 95% is one or two prompts.
- **Synthetic data.** Real call transcripts and customer messages are noisier than these.
- **One run on free, shared servers.** Latency from free tiers varies with load, so p95 is indicative only; a paid, dedicated deployment would be faster and steadier.

### Results and findings

| Category | Always cheapest | Always frontier | Router |
|---|---|---|---|
| Classification, extraction, document Q&A, summaries | 5.00 | 5.00 | 5.00 |
| Reasoning | 3.83 | 4.83 | 4.83 |
| Indian-language tasks | 4.40 | 4.60 | 4.30 |
| Translation | 4.20 | 3.80 | 4.20 |

**The hardest tenth.** These are the four test prompts where the cheapest model scored lowest. The router matched the frontier model's quality on all four, and was far cheaper on two:

| Prompt | Task | Cheapest | Frontier | Router | Router cost vs frontier |
|---|---|---|---|---|---|
| T37 | Tax and loan prepayment decision | 1 | 5 | 5 | ₹0.27 vs ₹0.27 (sent to frontier) |
| T28 | Hinglish UPI problem | 2 | 4 | 4 | ₹0.10 vs ₹0.30 |
| T36 | EMI calculation | 2 | 5 | 5 | ₹0.15 vs ₹0.15 (sent to frontier) |
| T18 | Hindi translation | 3 | 4 | 4 | ₹0.005 vs ₹0.04 |

**What stood out:**
1. **35 of 40 prompts (87.5%) never needed the frontier model,** and 85% of router responses came back in under 3 seconds.
2. **The frontier model is not automatically better at Indian languages.** The 550B model scored *lower* than the 20B model on Kannada translation (2 vs 4) and Tamil translation (3 vs 4). This is the case for a dedicated Indian-language tier, and for Sarvam's own Indic models.
3. **The slow tail stays slow.** The router's p95 (28.5 s) is close to the frontier's (32 s), because the slowest 5% of requests are the hard ones that still go to the frontier model. Routing cuts cost much more than it cuts tail latency.
4. **"Always cheapest" isn't as cheap as it looks.** On T37 the cheap model used its whole budget thinking and returned nothing. The fallback made T37 the most expensive prompt in that run.
5. **Indian-language answers use more tokens.** On the same model, a short Bengali answer (P10) cost more than 10x a short English document answer (P08).

**What the router gets wrong.** The keyword rules made three mistakes on the test set. I left them in rather than tune on test data:
- **T38 and T39,** multi-step maths with no trigger words, went to Tier 1. The cheap model got both right this time, but nothing in the rules would have caught it if it hadn't.
- **T14,** a routine summary, went to Tier 3 because the word "comparison" appears inside the transcript. That cost about 7x what the other summaries cost on Tier 2.

These are the limits of keyword rules: they cannot see difficulty that isn't spelled out. A classifier that gives a calibrated confidence, such as Sarvam Prism, can move uncertain requests up a tier.

**Indian-language tasks scored 4.30, below always-cheapest (4.40).** Two prompts sent to Tier 2 lost points: T23 (Hindi) ignored a four-sentence limit and wrote six, and on T27 (Bengali) the judge flagged spelling errors. On this set, sending Indian-language prompts to the larger gpt-oss model didn't beat the small one. That's the case for a dedicated Indian-language model in Tier 2, such as Sarvam 105B. Each configuration ran once, so a gap of one or two prompts may be noise.

### What I took from Cursor's and Ramp's routers

**Applied in this build:**
- **Explain every decision.** Each request logs its route, the reason, cost, latency and any fallback. Cursor's own users have asked to see which model handled each step; a bank's audit team would insist on it.
- **Prove the saving with a quality test.** Ramp tests models on its own production tasks before routing to them. I compared the router with both baselines on held-out prompts and report where it loses.

**What I'd add for a customer:**
- **Let the customer choose the trade-off, not the model.** Cursor offers Cost, Balance and Intelligence modes. Here, that would be a setting per workload that changes which tier each difficulty level goes to; I tested only the balanced setting. Unlike Cursor, I'd also let a bank pin a high-stakes workload to one model.
- **Measure cost per finished piece of work.** Cursor reports cost per commit, not per token. For a bank, that means cost per resolved call or per processed claim.
- **Learn from real use, and route around failures.** Cursor's router was trained on whether users had to stop and correct its results; Ramp's gateway moves traffic away from slow or failing providers as it happens. Both need live traffic, so they come after a pilot.

### Trade-offs considered and dropped

- **Send everything to the cheap model first, and move up a tier only if the answer is poor (a "cascade").** It needs no classifier, but hard prompts pay twice: a wasted cheap attempt, then the real one, in both cost and waiting time. It also needs a live quality check on every answer, an extra model call before the customer sees anything, to decide whether to move up. My router picks the tier up front instead, and moves up a tier only on obvious failures (a timeout, error, refusal or blank answer), which need no quality check. That saves the cost and waiting time of both the wasted first attempt and the check.
- **A circuit breaker on top of the timeout.** The timeout gives up on one slow request after 90 seconds and retries it one tier up. A circuit breaker would also stop sending new requests to a model that has failed several times recently, so they skip straight to the next tier instead of each waiting to fail first. The timeout is needed either way, because timeouts are how a breaker knows a model is struggling. I left the breaker out for three reasons. With one model per tier, its only move is to send traffic one tier up, which the fallback already does, so it would save waiting time during an outage without changing where requests end up. On free tiers, many failures are quota limits, which the router handles by waiting, so a breaker would often trip on quota and push traffic up for no reason. And the test sends one request at a time, so there is no queue of users for it to protect. The results bore this out: no rate limits, and the one fallback came from a token budget, not a failing model. In production, where an outage would hit many users at once, I'd add it with a cap on how long it can keep traffic on the pricier tier. If Tier 2 went down, every request would move to Tier 3 at about four times the price, so the team should be alerted after, say, 15 minutes, before an outage can quietly balloon the bill.
- **Using a model to classify each prompt, instead of keyword rules.** There were two options: ask a small chat model whether the request is simple, moderate or complex, or use Sarvam Prism. Both understand meaning rather than matching keywords, but both add an API call to every request, including the easy ones. A chat model also doesn't always reply in the requested format, and it can't say how sure it is in a way you can rely on, so there's no safe "move up if unsure" rule. The bigger problem is cost. The classifier has to read every prompt, so its cost grows with the customer's whole input, not just the hard requests. For a customer who sends far more text in than the model writes back, as with call summaries or document questions, a chat-model classifier would eat a large share of the saving that routing creates, and Prism a smaller but real share, while both add waiting time to every request. Keyword rules cost nothing, decide almost instantly, and every decision can be read and audited.
- **A custom-trained classifier.** The most accurate in principle, but it needs thousands of labelled customer requests. It's the right next step once a customer's logs exist.
- **Choosing the models, and a Tier 2 from a different family.** I wanted Sarvam 105B, an Indian-language specialist, as Tier 2, but no Sarvam credit arrived. On 14 Nov 2026 I tested 15 free models with the same Hindi question (`python router.py --probe`):
  - Google's Gemini 3.8 Flash and Gemma 4 returned repeated 500 and 503 errors or timed out on the free tier.
  - Several NVIDIA models (Kimi K3.1 Pro, Kimi K2.6, Nemotron 3 Super) were listed but returned "not found".
  - Others (DeepSeek V4.1 Flash, Mistral Large 4, MiniMax M3.1, GLM-5.3 Flash) didn't reply within 40 s.

  I kept only models that answered reliably, which left gpt-oss-120b, the larger sibling of Tier 1, as Tier 2. The upside is a clean comparison: the two tiers share training, tokenizer and provider, so the difference between them is mainly size. The cost is shared weaknesses: on this test, Tier 2 didn't beat Tier 1 on Indian-language tasks, and one provider outage would take out both tiers. On Sarvam's platform, Tier 2 would be Sarvam 105B.

### Scaling inside a customer VPC or an air-gapped site

A VPC is a private section of a cloud, such as the bank's own AWS account, where traffic never crosses the public internet. An air-gapped site has no internet connection at all.

- **Same code, local endpoints.** The router talks to every model through the same OpenAI-style API. Inside a VPC, each tier runs on the customer's own GPUs behind a standard inference server such as vLLM or NVIDIA NIM; the only change is the address in `config.py`.
- **Every tier can be self-hosted.** All three are open-weight models. gpt-oss-20b fits in 16 GB of GPU memory, gpt-oss-120b on one 80 GB GPU, and Nemotron 3 Ultra needs a multi-GPU server. In an air-gapped site, Tier 3 is the largest open-weight model the customer can host, and Sarvam 105B can take Tier 2.
- **Size each tier from the traffic mix.** The router's log shows how much work each tier does. In my test, Tier 3 answered 12% of requests but generated 40% of the output tokens, so it needs capacity for long, hard answers, not just request count.
- **Route by capacity as well as difficulty.** In the cloud, extra capacity is the provider's problem. On owned GPUs, Tier 3 has a fixed number of slots: when it is full, the router queues the request or, for workloads the customer allows, sends it to Tier 2 and logs the downgrade. With no outside provider, a failed tier can only fall back to another tier inside the site.
- **Scale the router by running more copies.** It holds no state and decides in about a millisecond, so copies sit behind the customer's load balancer. The GPUs, not the router, set the limit.
- **The customer controls updates.** A model changes only when the customer approves, after the new version passes the customer's own test set. Models, routing rules and thresholds arrive as signed, versioned bundles, so the customer's IT team can check where they came from before installing them.
- **Quality checks and logs stay inside.** The judge model and the test set run locally, so no prompt or answer leaves the site; only summary numbers are shared, if anything.
- **The economics change shape.** On owned GPUs, the hardware is already paid for. Routing sends most traffic to small models, so the same GPUs serve more requests and the next hardware purchase comes later.

---

## Part 2: The economics

The customer sends **50M input tokens and 10M output tokens a month.** Cost = input tokens ÷ 1M × input price + output tokens ÷ 1M × output price.

### Assumptions

| Assumption | Value | Source |
|---|---|---|
| Exchange rate | ₹94.20 per US$ | RBI reference rate, 13 Nov 2026 |
| Tier prices | As in the Stack table above | List prices, checked 14 Nov 2026 (see Price sources) |
| Traffic split | Input tokens: T1 22%, T2 64%, T3 13%. Output tokens: T1 21%, T2 39%, T3 40%. | Measured on the router's test run (`results.csv`) |
| Fallbacks | 0% (measured), with 2.5% as a sensitivity | Router run; cheapest baseline |
| Router cost | ₹0 (rules run locally); Prism would add ₹4 per 1M input tokens, or ₹200 a month | Sarvam price list |
| Caching and volume discounts | Ignored, which is conservative | – |
| Judge cost | Excluded: a measurement cost, not a serving cost | – |

### The four numbers

| | Input cost | Output cost | Monthly total |
|---|---|---|---|
| **1. Always cheapest** | 50M × ₹7.06 = ₹353 | 10M × ₹28.26 = ₹283 | **₹636** |
| **2. Always frontier** | 50M × ₹56.52 = ₹2,826 | 10M × ₹254.34 = ₹2,543 | **₹5,369** |
| **3. Router** | ₹906 | ₹1,297 | **₹2,203** |
| **4. Saving, router vs frontier** | | | **59%** (57.5% with 2.5% fallbacks) |

The saving here (59%) is higher than in the test (52%). This customer sends five input tokens per output token, and only 13% of input tokens needed the frontier model, against 40% of output tokens. Always-cheapest is cheapest, but its quality is why nobody runs it: it failed the hardest tenth (scores of 1, 2, 2 and 3).

### What the saving depends on

The saving depends almost entirely on how much traffic still needs the frontier model:

| Share of tokens still on the frontier model | Router monthly cost | Saving vs always-frontier |
|---|---|---|
| 5% | ₹1,298 | 76% |
| 15% | ₹1,726 | 68% |
| 30% | ₹2,369 | 56% |
| 40% | ₹2,798 | 48% |

### The same traffic on other price lists

| Scenario | Always frontier | Router | Saving |
|---|---|---|---|
| This build (frontier: Nemotron 3 Ultra) | ₹5,369 | ₹2,203 | 59% |
| Sarvam's own price list (Tier 2: Sarvam 105B at ₹15 / ₹60; Tier 3: GLM-5.3 at ₹126 / ₹390; router: Prism) | ₹10,200 | ₹3,444 | 66% |
| An illustrative closed frontier price of $5 / $25 per 1M (an assumption, not a specific model) | ₹47,100 | ₹13,318 | 72% |

The Sarvam row assumes Sarvam 105B performs at least as well as gpt-oss-120b; I didn't measure that. The pattern is clear, though: the more expensive the customer's default model, the more routing is worth. At this customer's volume, the saving is about ₹3,200 a month, which is small. For a ₹2 crore-a-year account running entirely on a frontier model, the same mix would save roughly ₹1.2 crore a year.

---

## Part 3: The business questions

*Note: the brief lists questions 1–5, 7 and 8; there is no question 6.*

### Q1. Pricing a ₹2 crore-a-year BFSI account

I'd pick a blended per-token rate, set per workload and sold as an annual commitment. Pass-through with a margin ties Sarvam's revenue to its cost, so every rupee the router saves is a rupee Sarvam loses. A blended rate sits between our cost to serve at the measured traffic mix and what the customer pays today, so the saving is shared: the bank pays less and Sarvam's margin grows. A flat platform subscription is predictable, but inference costs rise with usage, so heavy users erode the margin and light users feel overcharged at renewal. Outcome pricing sounds appealing, but every bill would hinge on agreeing what counted as a success and whether the model caused it, which is hard to audit and easy to dispute.

### Q2. Routing reduces billed tokens. Should we build it anyway?

Build it. If you don’t route, someone else will. Once the customer sees that most requests don’t need the frontier model, they’ll move that traffic to a cheaper competitor and keep only the hard requests with you. Routing keeps all the traffic on Sarvam. You risk losing a significant portion of the account if you don't build.

### Q3. 55% cheaper, +120 ms p95, 3% worse on the hardest tenth: for whom is this wrong?

It's wrong for two kinds of customer: real-time voice agents, where the model gets only a few hundred milliseconds per reply, so 120 ms at p95 is a large slice; and high-stakes decisions (credit underwriting, claims, fraud review, regulatory replies), where the hardest tenth carries the risk and one wrong answer costs more than the saving. In the contract I'd put an SLA with two promises per workload: a p95 latency cap, so the router is never too slow, and a quality floor measured monthly on the customer's own hard cases, so it never trades accuracy for savings. Missing either earns service credits. I'd add the right to pin any workload to the frontier model, and full routing logs for audit.

### Q4. "We'll just send everything to the frontier model"

"That's the right call while volume is small, because the bill is modest and one model is simpler to run. At production scale, though, most calls are classification, extraction and Indian-language replies, where a cheaper model does as well and sometimes better. Let's run your own traffic through both in shadow mode for two weeks; you switch only the workloads where quality is provably equal, and you keep a one-click override."

**First discovery question:** "What will your monthly model bill be once this runs on all your traffic, not just the pilot?" It tests whether cost will ever become real pain, and reveals their growth plan.

### Q5. Where Sarvam wins and loses against OpenRouter, Not Diamond, Martian and the hyperscalers

Sarvam wins where owning the models and the deployment matters. Its models are built for Indian languages, while most routers are tuned on English. It can run the whole routing stack on-premise or air-gapped, where hosted routers can't go. And because it owns the models, it controls their price: it can include routing free and discount where resellers, who pay list price to the model makers, can't. It loses on breadth: OpenRouter offers 500+ models, and Sarvam can't route to the strongest closed models. It also loses on neutrality, since a model maker routing to its own models looks conflicted, and on distribution, because banks already have hyperscaler contracts and security approvals.

### Q7. One Indian segment: first workload, week-two proof, path to ₹10 crore

Segment: lenders' collections and service call centres, where calls run in a dozen languages and Sarvam already has references such as Mahindra Finance. First workload: post-call summaries and outcome tags. It isn't latency-sensitive, most calls are simple enough for a small model, and QA teams already score calls by hand, so a benchmark exists on day one. Week-two proof, on past calls in shadow mode: tags match agents' own tags, QA rates a sample of summaries as good as today's, cost per call falls, and no data leaves the customer's environment. Path to ₹10 crore: summaries alone earn paise per call, so expand to transcribing every call, then collections voice agents, across five or six large lenders, sold on one rate card with one case study backed by hard numbers.

### Q8. Quality quietly degrades on the most valuable 5% of a customer's queries

The failure isn't the drop; it's that the customer found it first. Now: call them before they escalate, pin those queries to the frontier model at Sarvam's cost, use the routing logs to find the cause (a model update, a misroute or new kinds of queries), and send a written incident report with service credits they didn't have to ask for. From now on: turn their most valuable queries into a test set that is rerun on every model or routing change and refreshed monthly from live traffic, send a small sample of that traffic to the frontier model in the background to compare answers, and watch user signals such as repeat questions and escalations. Review that 5% with them weekly for three months, then monthly, so they see us catch problems first.

---

## Limitations and what I'd do next

- Swap in **Sarvam 105B and Prism** once credit is available, and rerun the same test.
- Replace keyword rules with a **classifier trained on a customer's own logs**, or Prism's calibrated confidence, to catch hard prompts that don't use obvious keywords.
- Add **native-speaker review** for Tamil, Bengali, Kannada and Marathi, and a second judge to check the first.
- Treat a **cut-off answer as a failure.** When a model hits its token cap, the provider flags it (`finish_reason = length` in the log), but the router currently accepts the answer. Counting it as a failure, and sending a request that ran out of tokens while thinking straight to Tier 3, would have rescued T37 in the always-cheapest run.
- Run **several passes on paid, dedicated endpoints** for a reliable p95, and include caching in the economics.
- Grow the test set to **hundreds of real, anonymised prompts** labelled by someone other than the router's author.

## Assumptions

- **Deadline:** the brief says one week at the top and "the 24-hour window" in the notes. I assumed one week.
- **Prices:** all prices were checked on 14 Nov 2026 and change often. Gemini prices, used only in the probe, double on 1 Jan 2027.

## How to run

```
python -m pip install -r requirements.txt
# put GROQ_API_KEY and NVIDIA_API_KEY in a .env file (never committed)
python router.py --check                                   # every model answers once
python router.py --config always_cheapest --set test
python router.py --config always_frontier --set test
python router.py --config router --set test
python router.py --summary                                 # writes summary.csv
python router.py --fallback-test                           # forces a fallback on 3 prompts
```

**Files:**
- `router.py`: the script.
- `config.py`: models, prices, thresholds.
- `prompts.csv`: the test set.
- `results.csv`: one row per prompt per configuration.
- `summary.csv`: one row per configuration.
- `fallback_test.csv`: the forced-fallback test.
- `api_calls_log.csv`: every live API call from the runs and checks (the `--probe` test isn't logged).
- `cache.jsonl` and `judge_cache.jsonl`: saved answers and scores.
