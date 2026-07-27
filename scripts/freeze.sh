#!/usr/bin/env bash
# R30 — code and version freeze.
#
# Records everything needed to rebuild and re-run the experiment exactly as it
# was at the freeze point, and creates the annotated git tag that names it:
#
#   commit.txt              frozen commit hash, branch, commit date, tree state
#   requirements.lock.txt   dependency lock (`pip freeze`, the project has no lock file)
#   pyproject.toml          the declared dependency ranges the lock resolves
#   schema.sql              database schema dump (pg_dump, or ORM DDL fallback)
#   RUN_INSTRUCTIONS.md     how to reproduce the run from the tag + the lock
#   freeze_manifest.json    final manifest referencing the frozen commit + lock (R30.2)
#   checksums.json          sha256 of every file above (shared R16 checksum code)
#
# Usage:
#   bash scripts/freeze.sh --tag v1.0.0 --dry-run      # rehearse, creates no tag
#   bash scripts/freeze.sh --tag v1.0.0                # create the annotated tag locally
#   bash scripts/freeze.sh --tag v1.0.0 --push         # ... and push it (opt-in)
#
# Repository state:
#   The ONLY mutation this script makes to the repository is the annotated tag,
#   and only when --dry-run is absent. It NEVER pushes unless --push is passed,
#   and it never commits, amends, checks out or moves a branch. Pushing a `v*`
#   tag is what triggers the release flow in .github/workflows/ci.yml: the
#   `release` job needs the `gate` job, so the tag only becomes a published
#   release once lint, type-check, tests/coverage, data-quality and the smoke
#   evaluation all pass (R29.2). Freeze locally, verify, then push.
#
# The freeze bundle is written OUTSIDE the tagged tree (artifacts/freeze/<tag> by
# default): it *describes* the frozen commit, so it cannot be part of it. Attach
# it to the release or commit it afterwards.
#
# Conventions follow scripts/pg_local.sh: plain bash, env-overridable defaults,
# no dependency on the venv being on PATH.

set -uo pipefail

TAG=""
MESSAGE=""
OUT_DIR=""
REMOTE="${FREEZE_REMOTE:-origin}"
DRY_RUN=0
ALLOW_DIRTY=0
REQUIRE_PG_DUMP=0
PUSH=0

usage() {
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Options:
  --tag NAME          tag to create (required), e.g. v1.0.0
  --message TEXT      annotation message (default: generated, cites commit + lock)
  --out-dir DIR       freeze bundle directory (default: artifacts/freeze/<tag>)
  --dry-run           record the bundle but do NOT create the tag
  --allow-dirty       proceed with uncommitted tracked changes (recorded as dirty)
  --require-pg-dump   fail instead of falling back when pg_dump/DATABASE_URL is unusable
  --push              push the tag to the remote after creating it (opt-in)
  --remote NAME       remote for --push (default: origin)
  -h, --help          this help
USAGE
}

die() {
  echo "freeze: $*" >&2
  exit 1
}

log() {
  echo "==> $*"
}

warn() {
  echo "    warning: $*" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) TAG="${2:-}"; shift 2 ;;
    --message) MESSAGE="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --remote) REMOTE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --require-pg-dump) REQUIRE_PG_DUMP=1; shift ;;
    --push) PUSH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

[ -n "$TAG" ] || die "--tag is required (e.g. --tag v1.0.0)"
if [ "$DRY_RUN" -eq 1 ] && [ "$PUSH" -eq 1 ]; then
  die "--push and --dry-run are mutually exclusive"
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
cd "$REPO_ROOT" || die "cannot enter repository root"

# Interpreter used for the dependency lock and the manifest helper. The venv is
# not assumed to be on PATH (Windows and POSIX layouts both handled).
if [ -z "${PYTHON:-}" ]; then
  for candidate in .venv/bin/python .venv/Scripts/python.exe python3 python; do
    if [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
[ -n "${PYTHON:-}" ] || die "no python interpreter found (set PYTHON=...)"
"$PYTHON" -c "import jobrec" >/dev/null 2>&1 ||
  die "cannot import jobrec with '$PYTHON' - install the project first (make install)"

git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null 2>&1 &&
  die "tag ${TAG} already exists; delete it or choose another name"

# Cleanliness is judged on *tracked* changes only: the tag captures the tracked
# tree, and untracked files (including this script's own output directory) do not
# change what the tag points at. Untracked files are still counted and reported.
DIRTY_TRACKED="$(git status --porcelain --untracked-files=no)"
UNTRACKED_COUNT="$(git ls-files --others --exclude-standard | wc -l | tr -d ' ')"
TREE_STATE="clean"
if [ -n "$DIRTY_TRACKED" ]; then
  TREE_STATE="dirty"
  if [ "$ALLOW_DIRTY" -eq 1 ]; then
    warn "working tree has uncommitted tracked changes; freezing anyway (--allow-dirty)"
  else
    die "working tree has uncommitted tracked changes - commit them, or pass --allow-dirty
$(echo "$DIRTY_TRACKED" | sed 's/^/       /')"
  fi
fi

COMMIT="$(git rev-parse HEAD)" || die "cannot resolve HEAD"
COMMIT_SHORT="$(git rev-parse --short=12 HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
COMMITTED_AT="$(git log -1 --format=%cI)"
SUBJECT="$(git log -1 --format=%s)"

OUT_DIR="${OUT_DIR:-artifacts/freeze/${TAG}}"
mkdir -p "$OUT_DIR" || die "cannot create ${OUT_DIR}"

log "freezing ${TAG} at ${COMMIT_SHORT} (branch ${BRANCH}, tree ${TREE_STATE})"
log "bundle: ${OUT_DIR}"

# ---------------------------------------------------------------- commit record
{
  echo "tag=${TAG}"
  echo "commit=${COMMIT}"
  echo "commit_short=${COMMIT_SHORT}"
  echo "branch=${BRANCH}"
  echo "committed_at=${COMMITTED_AT}"
  echo "subject=${SUBJECT}"
  echo "worktree=${TREE_STATE}"
  echo "untracked_files=${UNTRACKED_COUNT}"
  if [ -n "$DIRTY_TRACKED" ]; then
    echo "uncommitted_tracked_changes:"
    echo "$DIRTY_TRACKED"
  fi
} >"${OUT_DIR}/commit.txt"

# ------------------------------------------------------------- dependency lock
# The project ships pyproject.toml with version ranges and no lock file, so the
# practical lock is a resolved `pip freeze` of the environment the experiment ran
# in. --exclude-editable drops the editable install of the project itself, which
# is pinned by the tag (and whose path is machine-specific).
log "recording dependency lock (pip freeze) -> ${OUT_DIR}/requirements.lock.txt"
"$PYTHON" -m pip freeze --exclude-editable >"${OUT_DIR}/requirements.lock.txt" ||
  die "pip freeze failed"
LOCK_LINES="$(wc -l <"${OUT_DIR}/requirements.lock.txt" | tr -d ' ')"
[ "$LOCK_LINES" -gt 0 ] || die "pip freeze produced an empty lock"
cp pyproject.toml "${OUT_DIR}/pyproject.toml" || die "cannot copy pyproject.toml"

# --------------------------------------------------------------- schema record
# Preferred: a real `pg_dump --schema-only` of the configured database.
# Fallback: the ORM metadata compiled for the PostgreSQL dialect, which needs
# neither pg_dump nor a running server. The freeze is never blocked by a missing
# tool unless --require-pg-dump asks for exactly that.
SCHEMA_FILE="schema.sql"
SCHEMA_SOURCE=""
SCHEMA_NOTE=""

DB_URL="${DATABASE_URL:-}"
if [ -z "$DB_URL" ] && [ -f .env ]; then
  # Value only, never echoed: it may carry a password.
  DB_URL="$(sed -n 's/^DATABASE_URL=//p' .env | tail -n1 | tr -d '\r' | sed 's/^["'"'"']//;s/["'"'"']$//')"
fi
# pg_dump speaks libpq URLs; strip any SQLAlchemy driver suffix (+psycopg).
PG_URL="$(printf '%s' "$DB_URL" | sed 's|^postgresql+[a-z0-9]*://|postgresql://|')"

if command -v pg_dump >/dev/null 2>&1 && [ -n "$PG_URL" ]; then
  log "dumping database schema (pg_dump --schema-only)"
  if pg_dump --schema-only --no-owner --no-privileges --dbname="$PG_URL" \
      >"${OUT_DIR}/${SCHEMA_FILE}" 2>"${OUT_DIR}/.pg_dump.err"; then
    SCHEMA_SOURCE="pg_dump"
  else
    warn "pg_dump failed: $(tr -d '\r' <"${OUT_DIR}/.pg_dump.err" | tail -n1)"
    rm -f "${OUT_DIR}/${SCHEMA_FILE}"
  fi
  rm -f "${OUT_DIR}/.pg_dump.err"
elif ! command -v pg_dump >/dev/null 2>&1; then
  warn "pg_dump not on PATH"
else
  warn "DATABASE_URL is not set (and none found in .env)"
fi

if [ -z "$SCHEMA_SOURCE" ]; then
  if [ "$REQUIRE_PG_DUMP" -eq 1 ]; then
    die "--require-pg-dump was passed but pg_dump could not produce a schema dump"
  fi
  log "falling back to the ORM schema DDL (no live database required)"
  if "$PYTHON" scripts/freeze_record.py schema-ddl >"${OUT_DIR}/${SCHEMA_FILE}"; then
    SCHEMA_SOURCE="sqlalchemy-metadata"
    SCHEMA_NOTE="pg_dump unavailable; DDL rendered from jobrec.storage.models for the PostgreSQL dialect"
  else
    rm -f "${OUT_DIR}/${SCHEMA_FILE}"
    SCHEMA_SOURCE="skipped"
    SCHEMA_NOTE="no schema could be recorded: pg_dump unavailable and the ORM DDL fallback failed"
    warn "$SCHEMA_NOTE"
  fi
fi
log "schema source: ${SCHEMA_SOURCE}"

# ------------------------------------------------------------ run instructions
log "writing run instructions -> ${OUT_DIR}/RUN_INSTRUCTIONS.md"
sed -e "s|__TAG__|${TAG}|g" \
    -e "s|__COMMIT__|${COMMIT}|g" \
    -e "s|__SCHEMA_SOURCE__|${SCHEMA_SOURCE}|g" \
    >"${OUT_DIR}/RUN_INSTRUCTIONS.md" <<'FREEZE_DOC'
# Reproducing the frozen CMJCC experiment (__TAG__)

Frozen commit: `__COMMIT__`
Dependency lock: `requirements.lock.txt` (resolved `pip freeze`)
Database schema: `schema.sql` (source: __SCHEMA_SOURCE__)

Verify this bundle first (any tampering shows up as a mismatch):

    python -c "import json,pathlib,hashlib; d=pathlib.Path('.'); m=json.loads((d/'checksums.json').read_text()); bad=[p for p,h in m.items() if hashlib.sha256((d/p).read_bytes()).hexdigest()!=h]; print('MISMATCH', bad) if bad else print('checksums OK')"

## 1. Check out the frozen code

    git fetch --tags
    git checkout __TAG__            # or: git checkout __COMMIT__

## 2. Recreate the environment from the lock

    python -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.lock.txt
    .venv/bin/pip install -e . --no-deps

`--no-deps` keeps the locked versions exactly as recorded; the project itself is
installed from the checked-out tag.

## 3. Database (optional)

The recommendation pipeline runs without a database. For the persisted runs:

    docker compose -f docker/docker-compose.yml up -d postgres
    export DATABASE_URL=postgresql+psycopg://jobrec:jobrec@localhost:5432/jobrec

`schema.sql` is the schema of record for this freeze. The application creates the
tables itself (`jobrec.storage.db.create_all`, which also stamps the migration
version), so restoring the dump by hand is only needed to inspect the schema:

    psql "$DATABASE_URL" -f schema.sql

## 4. Data and experiment

    make prepare-data
    make test
    python -m jobrec_eval.cli pipeline \
      --config configs/experiment_full.yaml \
      --scenarios evaluation/data/scenarios.jsonl \
      --catalog data/processed/jobs.jsonl \
      --out-root evaluation/outputs

Every produced bundle carries its own `checksums.json`; verify it with:

    python -m jobrec_eval.cli verify evaluation/outputs/exp-<id>

## 5. Release

Pushing the `__TAG__` tag runs the CI gate (lint, type-check, tests + coverage,
data quality, deterministic smoke evaluation). The release for the tag is only
published after that gate is green.
FREEZE_DOC

# ------------------------------------------------------------------- the tag
TAG_STATE="not-created"
if [ -z "$MESSAGE" ]; then
  MESSAGE="CMJCC freeze ${TAG}

commit: ${COMMIT}
dependency lock: requirements.lock.txt (${LOCK_LINES} pinned packages)
schema: ${SCHEMA_FILE} (${SCHEMA_SOURCE})
freeze bundle: ${OUT_DIR}"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  log "dry run: NOT creating a tag. Would run:"
  echo "    git tag -a ${TAG} ${COMMIT_SHORT} -m <message>"
  echo "    message:"
  echo "$MESSAGE" | sed 's/^/      /'
else
  log "creating annotated tag ${TAG}"
  git tag -a "$TAG" "$COMMIT" -m "$MESSAGE" || die "git tag failed"
  TAG_STATE="created"
fi

PUSHED_FLAG=""
if [ "$PUSH" -eq 1 ]; then
  log "pushing ${TAG} to ${REMOTE} (this triggers the CI gate + release flow)"
  git push "$REMOTE" "refs/tags/${TAG}" || die "git push failed; the local tag is kept"
  PUSHED_FLAG="--pushed"
fi

# ---------------------------------------------------------------- the manifest
log "writing the freeze manifest"
# shellcheck disable=SC2086  # PUSHED_FLAG is an intentional optional flag
"$PYTHON" scripts/freeze_record.py manifest \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --commit "$COMMIT" \
  --branch "$BRANCH" \
  --committed-at "$COMMITTED_AT" \
  --tree-state "$TREE_STATE" \
  --tag-state "$TAG_STATE" \
  --schema-source "$SCHEMA_SOURCE" \
  --schema-file "$SCHEMA_FILE" \
  --schema-note "$SCHEMA_NOTE" \
  $PUSHED_FLAG >/dev/null || die "writing the freeze manifest failed"

log "freeze bundle complete:"
ls -1 "$OUT_DIR" | sed 's/^/    /'
if [ "$DRY_RUN" -eq 1 ]; then
  log "dry run finished: no tag was created, nothing was pushed"
elif [ "$PUSH" -eq 0 ]; then
  log "tag ${TAG} exists locally only. Push it when you are ready:"
  echo "    git push ${REMOTE} refs/tags/${TAG}"
fi
