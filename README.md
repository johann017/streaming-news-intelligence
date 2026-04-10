# Streaming News Intelligence

A production-grade, real-time global news aggregation and ML summarization platform.
Runs entirely within free tiers — GitHub Actions for compute, Firebase Firestore for storage.

> **Status:** Under active development. See the chunk plan for progress.

## Architecture

```
GitHub Actions (cron: */5 * * * *)
         │
         ▼
┌─────────────────────────────────────────────┐
│                  pipeline.py                │
│                                             │
│  ingestion → processing → clustering        │
│       → ranking → summarization             │
│       → firestore_writer → notifier         │
└─────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
   Firestore (events)       ntfy.sh (phone)
```

## Data Flow

```
RSS/Reddit/GDELT
      │
      ▼
  RawArticle  ──hash dedup──►  NormalizedArticle
                                      │
                               sentence-transformers
                                      │
                                    DBSCAN
                                      │
                                   Cluster
                                      │
                               score + rank
                                      │
                               distilbart summarize
                                      │
                                    Event
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                     ▼
              Firestore                            ntfy.sh push
```

## Setup

See [SETUP.md](SETUP.md) for full instructions once all chunks are complete.

## Cost

| Service | Free Tier | Our Usage |
|---|---|---|
| GitHub Actions | Unlimited (public repo) | ~2 min/run × 288 runs/day |
| Firebase Firestore | 1GB storage, 50k writes/day | <50MB, ~14k writes/day |
| ntfy.sh | Unlimited | <10 notifications/day |
| HuggingFace models | Free download | Ephemeral in Actions runner |

**Total cost: $0/month**

## License

MIT
