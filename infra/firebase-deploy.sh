#!/usr/bin/env bash
# Deploy Firestore rules and indexes to Firebase.
# Requires firebase-tools: npm install -g firebase-tools
# Run from the repo root after `firebase login`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Deploying Firestore rules and indexes..."
cd "${REPO_ROOT}/infra"
firebase deploy --only firestore --project "${FIREBASE_PROJECT_ID:-}"

echo "Done."
