# Sensitive Information Reference

This file documents everything in this project that is sensitive, private, or
should never be committed to the public repository. Review before each push.

---

## GitHub Actions Secrets (never put in code or config files)

| Secret name | What it is | Risk if exposed |
|---|---|---|
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Full JSON content of your Firebase service account key | Full Firestore read/write access, billing charges if storage or read quotas are abused |
| `FIREBASE_PROJECT_ID` | Your Firebase project ID | Low risk alone, but combined with credentials allows targeting your project |
| `NTFY_TOPIC` | Your private ntfy.sh topic name | Anyone who knows this string can send push notifications to your phone |
| `FIREBASE_TOKEN` | Firebase CLI auth token (used for Firestore rules deployment) | Can deploy/overwrite your Firestore rules |

**Where to set these:** GitHub repo → Settings → Secrets and variables → Actions

---

## Local Files (gitignored — never commit)

| File | What it contains | Already in .gitignore? |
|---|---|---|
| `.env` | All secrets for local development | ✅ Yes |
| `service-account*.json` / `firebase-adminsdk*.json` | Firebase service account key file | ✅ Yes |
| `data/cursors.json` | Ingestion state (not sensitive, but noisy) | ✅ Yes |
| `data/seen_ids.json` | Deduplication state (not sensitive) | ✅ Yes |
| `data/notified_ids.json` | Notification dedup state (not sensitive) | ✅ Yes |
| `data/embedding_cache.pkl` | Cached ML embeddings (large binary, ~100MB+) | ✅ Yes |
| `data/device_tokens.json` | FCM device tokens (if ever added) | ✅ Yes |

---

## ntfy.sh Topic Security

Your ntfy.sh topic acts as both the "address" and the "password" for
notifications. Anyone who knows the topic name can:
- Send you notifications (spam risk)
- Subscribe to see what notifications are being sent

**Recommendations:**
- Use a UUID or random 32-character string as your topic (e.g. `openssl rand -hex 16`)
- Never include the topic name in commit messages, PR descriptions, or issue comments
- If you suspect the topic is compromised, change it (update `NTFY_TOPIC` secret and your phone subscription)

---

## Firebase Service Account Key

The `FIREBASE_SERVICE_ACCOUNT_JSON` secret grants the pipeline Admin SDK access,
which **bypasses Firestore security rules**. This means it can read, write, and
delete any document in your Firestore database.

**Recommendations:**
- Never download this key to shared machines
- Rotate it annually (Firebase console → Project Settings → Service Accounts → Generate new private key)
- Delete old keys after rotation
- The key is only valid for YOUR Firebase project and has no access to anything else

---

## What is Safe to Make Public

Everything else in this repo is safe to be public:
- All Python source code
- Firestore security rules (`infra/firestore.rules`)
- GitHub Actions workflow files (they reference secrets by name, not value)
- `requirements.txt`, `pyproject.toml`
- `README.md`, `SENSITIVE.md` (this file)
- Test files

---

## Quick Security Checklist Before Pushing

- [ ] No API keys, tokens, or passwords in code
- [ ] `.env` is in `.gitignore` and not staged (`git status`)
- [ ] No service account JSON files in the repo directory
- [ ] `data/` files are gitignored (pipeline state, not source code)
- [ ] `NTFY_TOPIC` value is not visible in any committed file
