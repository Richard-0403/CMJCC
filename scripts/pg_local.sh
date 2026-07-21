#!/usr/bin/env bash
# Helper to (re)start a local PostgreSQL instance for development / verification
# inside sandboxes where a daemon cannot persist across separate shell sessions.
#
# Usage (run everything you need within a SINGLE shell invocation):
#   source scripts/pg_local.sh && pg_up && pytest tests/integration -k postgres
#
# Production deployments should use docker-compose (see docker/docker-compose.yml)
# instead of this script.
set -u

PGBIN="${PGBIN:-/usr/bin}"
PGDATA="${PGDATA:-/var/lib/pgsql/data}"
PGPORT="${PGPORT:-5432}"
PG_SOCKET_DIR="${PG_SOCKET_DIR:-/var/run/postgresql}"
DB_NAME="${DB_NAME:-jobrec}"
DB_USER="${DB_USER:-jobrec}"
DB_PASS="${DB_PASS:-jobrec}"

pg_up() {
  sudo -u postgres "${PGBIN}/pg_ctl" -D "${PGDATA}" -w \
    -o "-p ${PGPORT} -c unix_socket_directories='${PG_SOCKET_DIR}' -c listen_addresses='127.0.0.1'" \
    start || true
  sleep 2
  pg_isready -h 127.0.0.1 -p "${PGPORT}"
  sudo -u postgres psql -h 127.0.0.1 -p "${PGPORT}" -tc \
    "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 || \
    sudo -u postgres psql -h 127.0.0.1 -p "${PGPORT}" -c \
    "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}' CREATEDB;"
  sudo -u postgres psql -h 127.0.0.1 -p "${PGPORT}" -tc \
    "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 || \
    sudo -u postgres psql -h 127.0.0.1 -p "${PGPORT}" -c \
    "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"
  export DATABASE_URL="postgresql+psycopg://${DB_USER}:${DB_PASS}@127.0.0.1:${PGPORT}/${DB_NAME}"
  echo "DATABASE_URL=${DATABASE_URL}"
}

pg_down() {
  sudo -u postgres "${PGBIN}/pg_ctl" -D "${PGDATA}" -m fast stop || true
}
