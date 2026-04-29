# Streaming News Intelligence

A real-time global news aggregation and ML summarization platform that runs entirely within free tiers — GitHub Actions for compute, Firebase Firestore for storage.

Every 15 minutes, a pipeline fetches articles from RSS feeds, The Guardian API, and GDELT; clusters them by semantic similarity to detect which stories are being reported across multiple outlets; scores and ranks those clusters; summarizes the top ones with a transformer model; persists events to Firestore; and pushes phone notifications for the highest-impact stories.

---

## Architecture

```
GitHub Actions (cron: */15 * * * *)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                    pipeline.py                      │
│                                                     │
│  ingestion → processing → clustering → ranking      │
│       → summarization → firestore_write → notify   │
└─────────────────────────────────────────────────────┘
         │                            │
         ▼                            ▼
   Firestore (events)          ntfy.sh (phone)
```

**Stage error isolation:** ingestion, processing, clustering, and ranking are critical — a failure aborts the run. Summarization, Firestore writes, and notifications are non-critical — a failure is logged and the pipeline continues.

---

## Data Flow

```
RSS feeds + Guardian API + GDELT
          │
          ▼
      RawArticle  ──hash dedup──►  NormalizedArticle
                                          │
                                   all-MiniLM-L6-v2
                                   (sentence embedding)
                                          │
                                        DBSCAN
                                   (cosine distance)
                                          │
                                       Cluster
                                          │
                                   composite score
                                   (diversity/count/recency)
                                          │
                                         rank
                                          │
                                   facebook/bart-large-cnn
                                   (abstractive summary)
                                          │
                                         Event
                                          │
                        ┌────────────────┴───────────────────┐
                        ▼                                     ▼
                  Firestore                          ntfy.sh push
               (events collection)               (top events only)
```

---

## Ingestion Sources

| Source | What it provides |
|---|---|
| **RSS** | BBC World, Reuters, NYT World/US, Al Jazeera, NPR, AP |
| **Guardian API** | World, US, politics, business, science, environment, tech — with full body text |
| **GDELT** | Global event signals with geographic metadata |

Each source maintains a **cursor** (last-seen timestamp) so only new articles are fetched each run. Within a batch, articles are deduplicated by content hash. A global seen-IDs store prevents re-processing articles across runs.

---

## Pipeline Stages

### 1. Ingestion
Fetches from all three sources and merges into a single list of `RawArticle` objects, deduplicated by ID. Writes `data/raw_articles.json`.

### 2. Processing
- Filters articles shorter than 5 words or older than 24 hours
- Language detection (non-English filtered out)
- Normalizes whitespace and metadata
- Writes `data/normalized_articles.json`

### 3. Clustering
Embeds each article's title+body with `all-MiniLM-L6-v2` (embedding results cached across runs). Clusters embeddings with DBSCAN at `eps=0.12` cosine distance — tight enough that only articles about the same specific event cluster together. Noise points (unclustered articles) become singleton clusters. GDELT-only clusters are skipped in downstream stages. Writes `data/clusters.json`.

### 4. Ranking
Scores each cluster with a weighted composite:

| Component | Weight | Logic |
|---|---|---|
| Source diversity | 60% | `min(unique_sources / 3, 1.0)` — dominant signal |
| Article count | 25% | `log2(count) / 5` — diminishing returns |
| Recency | 15% | Exponential decay with configurable half-life |

Singleton clusters receive a 0.8× penalty. Top 50 clusters are kept. The top 10 by score are flagged as top events. Writes `data/ranked_clusters.json`.

### 5. Summarization
For each cluster above the score threshold (default 0.3), selects the best non-GDELT representative article and generates a 30–130 token abstractive summary using `facebook/bart-large-cnn` (~1.6 GB, cached across GitHub Actions runs). Falls back to extractive summarization (first 2 sentences) if the model fails. Writes `data/events.json`.

### 6. Firestore Write
Upserts events into the Firestore `events` collection using `set(merge=True)` — pure writes, no pre-fetch reads. Enforces a 500-document cap using a single COUNT aggregation query (1 read). Document expiry is handled server-side via Firestore TTL on the `expires_at` field.

### 7. Notify
Sends push notifications via ntfy.sh for any top event not previously notified. Deduplication is tracked in `data/notified_ids.json` (persisted in GitHub Actions cache). Up to 5 notifications per run. Priority is `urgent` for score ≥ 0.85, `high` otherwise.

---

## State Persistence

Between GitHub Actions runs, the following files are cached using `actions/cache`:

| File | Purpose |
|---|---|
| `data/cursors.json` | Per-source timestamps for incremental fetching |
| `data/seen_ids.json` | Global dedup — prevents reprocessing old articles |
| `data/notified_ids.json` | Prevents sending duplicate push notifications |
| `data/embedding_cache.pkl` | Cached sentence embeddings (avoids recomputing) |
| `~/.cache/huggingface` | HuggingFace model weights (~1.6 GB total) |

---

## Setup

### 1. Clone and configure

```bash
cp .env.example .env
# fill in your values
```

### 2. Required secrets (GitHub Actions)

| Secret | How to get it |
|---|---|
| `GUARDIAN_API_KEY` | Free at [open-platform.theguardian.com](https://open-platform.theguardian.com/access/) |
| `NTFY_TOPIC` | Choose a private topic name; install ntfy app on your phone |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Firebase console → Project settings → Service accounts |
| `FIREBASE_PROJECT_ID` | Your Firebase project ID |

### 3. Deploy Firestore rules and indexes

```bash
cd infra && bash firebase-deploy.sh
```

### 4. Run locally

```bash
pip install -r requirements.txt
python backend/workers/pipeline.py
```

---

## Cost

| Service | Free Tier | Our Usage |
|---|---|---|
| GitHub Actions | Unlimited (public repo) | ~3–4 min/run × 96 runs/day |
| Firebase Firestore | 1 GB storage, 50k reads/day, 20k writes/day | <50 MB, ~1 read + ~10 writes/run |
| ntfy.sh | Unlimited | <10 notifications/day |
| The Guardian API | 5,000 req/day | 96 req/day |
| HuggingFace models | Free download | Cached in Actions runner |

**Total cost: $0/month**

---

## Repository Structure

```
backend/workers/
  pipeline.py          # Orchestrator — sequences all stages
  firestore_writer.py  # Upserts events; enforces document cap
  firestore_client.py  # Firebase Admin SDK client singleton
  notifier.py          # ntfy.sh push notifications
  cleanup.py           # Manual Firestore maintenance (not in pipeline)

services/
  ingestion/
    rss_fetcher.py       # Parses RSS/Atom feeds
    guardian_fetcher.py  # Guardian Content API with cursor
    gdelt_fetcher.py     # GDELT event feed
    cursor_store.py      # Persists per-source fetch cursors
  processing/
    normalizer.py        # Cleans and normalizes articles
    deduplicator.py      # Seen-ID tracking across runs
    filters.py           # Age, length, language filters
  clustering/
    embedder.py          # sentence-transformers + embedding cache
    clusterer.py         # DBSCAN on cosine distance
    cluster_builder.py   # Assembles Cluster objects from groups
  ranking/
    scorer.py            # Composite score (diversity/count/recency)
    ranker.py            # Sorts and flags top events
  summarization/
    summarizer.py        # bart-large-cnn with extractive fallback
    event_builder.py     # Assembles Event objects
    key_points.py        # Bullet-point extraction

shared/
  models.py    # RawArticle, NormalizedArticle, Cluster, Event dataclasses
  config.py    # All config from environment variables
  utils.py     # Logging, hashing, text helpers

infra/
  firestore.rules         # Security rules (deny all client reads)
  firestore.indexes.json  # Composite indexes for cap enforcement
  firebase-deploy.sh      # Deploy rules + indexes

.github/workflows/
  pipeline.yml      # Main cron pipeline (*/15 * * * *)
  ci.yml            # Tests + linting on PRs
  deploy-infra.yml  # Firebase deployment workflow
```

---

## License

MIT
