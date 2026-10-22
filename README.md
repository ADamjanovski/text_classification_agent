# Text Classification Assistant

## Product 

Every day a newsroom receives a flood of incoming text — wire copy, headlines, story
snippets — and someone has to decide where each piece belongs before anyone can act on it.
Doing this by hand is slow, repetitive, and easy to get wrong when the queue is long.

This app is a **first-pass sorting assistant for news text**. A person pastes in a headline
or a short article and the app instantly suggests which of four topics it belongs to:

- **World** — international news, politics, conflict, and current affairs
- **Sports** — games, athletes, leagues, and competitions
- **Business** — companies, markets, money, and the economy
- **Sci/Tech** — science, technology, gadgets, and research

The app does not make the final call. It gives a fast, consistent first guess and, just as
importantly, **raises a hand when it is unsure** so a person knows exactly which items still
need a human eye. Confident, routine items can be sorted in bulk; the tricky or risky ones are
flagged for review.

**Why it is useful:**

- **Speed.** It turns "read every item and decide" into "review the suggestions and confirm,"
  which lets one person clear a much larger backlog.
- **Consistency.** The same kind of story gets sorted the same way every time, instead of
  depending on who happens to be on shift.
- **Honesty about uncertainty.** Instead of silently guessing, the app marks the borderline
  cases so people spend their attention where it actually matters.

**Who would use it:** a newsroom or content-triage analyst responsible for routing incoming
articles into the right topic queue.

---

## Dataset 

The app is built on **AG News**, a well-known, free, and publicly available collection of real
news articles. It is one of the most widely used datasets for news-topic classification, so it
is well understood and easy to reproduce.

- **Size:** about **127,600 articles** in total — **120,000** for teaching the app and
  **7,600** held back to test how well it learned.
- **What each article looks like:** a short headline plus a one- or two-sentence summary, which
  is exactly the kind of short text a triage analyst skims in real life.

**Category breakdown.** The dataset is split evenly across the same four topics the app
predicts, so no single category dominates:

| Category | Share | Training articles | Test articles |
|---|---|---|---|
| World | 25% | 30,000 | 1,900 |
| Sports | 25% | 30,000 | 1,900 |
| Business | 25% | 30,000 | 1,900 |
| Sci/Tech | 25% | 30,000 | 1,900 |

**Why it is a good fit:**

- **It matches the job.** The app's purpose is to sort news into topics, and this is a dataset
  of news already sorted into topics — the data and the task line up perfectly.
- **It is realistic.** These are genuine articles written by real journalists, full of the
  messy, varied language the app will see in production, not clean lab examples.
- **It is balanced and fair.** With an equal number of articles per category, the app learns all
  four topics equally well and is not biased toward whichever topic happened to be most common.
- **It is the right size.** Large enough for the app to learn reliable patterns, but small enough
  to train quickly on an ordinary laptop without special hardware.

**The value proposition in one line:** built on a large, balanced, real-world news dataset, the
app gives a newsroom a fast and consistent first guess at where each story belongs — and clearly
flags the ones a person should double-check — so analysts spend less time sorting and more time
on the decisions that need real judgment.

---


## How to run

Clone the repository and start the full stack:

```bash
docker compose up --build
```

The Streamlit frontend will be available at `http://localhost:8501` and the FastAPI backend at `http://localhost:8000`. MongoDB, the backend, and the frontend start in the correct order with health checks between services.

---

## Model summary

**TF-IDF + Logistic Regression**
- Accuracy: 91.83 % — Macro F1: 91.81 %
- Fit time: 34.6 s — Inference: ~49 µs/sample
- Uses word n-grams; fitted only on the training set. The primary model in the app.

**Transformer embeddings + Logistic Regression**
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Accuracy: 89.46 % — Macro F1: 89.45 %
- Fit time: ~60 min (encoding) — Inference: ~28 ms/sample
- Used as a second opinion alongside TF-IDF. Disagreement between the two models is itself a review signal.

TF-IDF is the stronger model for this dataset. Structured newswire text is keyword-rich, which is exactly where count-based methods excel. The transformer does not improve accuracy here and is 570× slower at inference, but it adds value as an independent signal for the review flag.

---

## Human-review rule

An article is flagged for manual review if any of the following conditions hold:

- Either model's confidence is below **40%**
- The margin between the top two predicted classes is below **5%**
- The two models **disagree** on the predicted label
- The text is **fewer than 50 characters** (too short to classify reliably)

When the flag fires, the note field explains the specific reason so the analyst knows what to look for.

---

## What works well

**Fast and accurate on typical news text.** For standard newswire articles with explicit topic keywords the TF-IDF model is correct more than 91 % of the time and returns a result in under a millisecond. The analyst can process a large queue quickly without reading each article.

**Reliable review flagging.** The confidence threshold and disagreement trigger correctly surface the genuinely hard cases: articles that sit on the Business/Sci-Tech boundary, Olympic coverage that the dataset labels as World but reads as Sports, and short or vague inputs with insufficient context. In every case analysed, flagged examples genuinely benefit from human review.

---

## What fails

**US-centric sports vocabulary.** The TF-IDF model's top Sports features are dominated by North American sports terms (baseball, league, NBA, stadium). Articles about cricket, rugby, or non-Olympic international sports are misclassified because those terms were never learned as strong Sports signals.

**Figurative headlines.** Both models break down when headlines use idiomatic or metaphorical language. "Virgin dips its toe into the Pacific" is a Business story about an airline route, but the embedding model predicts Sports. "Running may have defined the body" is a Sci/Tech study, but "running" pulls the prediction toward Sports. TF-IDF handles these better when the article body contains literal anchor words, but the transformer's semantic generalisation becomes a liability here.

**Label noise on the Business/Sci-Tech boundary.** The AG News labellers tagged articles by the identity of the main actor rather than the article's content. Stories about Google's IPO mechanics or Charles Schwab's pricing strategy are labelled Sci/Tech despite having no technical content. Both models predict Business and are arguably more correct than the ground-truth label. Any downstream system that relies on these predictions will inherit this labelling inconsistency.

---

## What to do next

- **Expand sports vocabulary.** Add international sports corpora or augment training data with cricket, rugby, and Formula 1 articles so the Sports class learns non-US vocabulary.
- **Add a metaphor/idiom detector.** A simple heuristic (e.g. flagging headlines with no literal topic word) would catch figurative inputs before they reach the classifier and route them directly to human review.
- **Audit and relabel the Business/Sci-Tech boundary.** A targeted relabelling pass on ~1 000 boundary examples would significantly reduce noise and improve model reliability on that pair. The current label inconsistency is the single biggest source of avoidable errors.

