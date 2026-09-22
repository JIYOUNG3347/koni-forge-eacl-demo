#!/usr/bin/env bash
# Start KONI-Forge Lite locally: Redis, the API server and the Celery worker.
#
#   ./run.sh          start everything and follow the logs
#   ./run.sh stop     stop the API server and the worker
#
# Requirements: python 3.11+, redis-server, and a built UI (npm --prefix UI run build:deploy).
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
PID_DIR=".run"
LOG_DIR="${LOG_DIR:-.run/logs}"

stop() {
    for name in api worker; do
        pid_file="$PID_DIR/$name.pid"
        if [ -f "$pid_file" ]; then
            pid="$(cat "$pid_file")"
            if kill -0 "$pid" 2>/dev/null; then
                echo "[run] stopping $name (pid $pid)"
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
}

if [ "${1:-start}" = "stop" ]; then
    stop
    exit 0
fi

if [ ! -x "$PYTHON" ]; then
    echo "[run] $PYTHON not found. Create the environment first:" >&2
    echo "      python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# .env is optional; anything already exported wins.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

export APP_HOST="${APP_HOST:-127.0.0.1}"
export APP_PORT="${APP_PORT:-8000}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://localhost:6379/1}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://localhost:6379/2}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

if [ -z "${INTERNAL_TOKEN:-}" ]; then
    # The worker calls back into the API with this token, so both need the same
    # value. Generating one per run is fine because they start together.
    INTERNAL_TOKEN="$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')"
    export INTERNAL_TOKEN
fi

if [ -z "${KONI_ADMIN_PW:-}" ]; then
    echo "[run] KONI_ADMIN_PW is not set. Copy .env.example to .env and set it." >&2
    exit 1
fi

mkdir -p "$PID_DIR" "$LOG_DIR"
stop

# ── Redis ────────────────────────────────────────────────
if ! redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
    if ! command -v redis-server >/dev/null 2>&1; then
        echo "[run] redis-server not found. Install it (brew install redis) or start Redis yourself." >&2
        exit 1
    fi
    echo "[run] starting redis-server"
    redis-server --port 6379 --daemonize yes --save '' --appendonly no
    sleep 1
fi
echo "[run] redis is up"

# ── API server ───────────────────────────────────────────
"$PYTHON" -m uvicorn api:app --host "$APP_HOST" --port "$APP_PORT" --workers 1 \
    >"$LOG_DIR/api.log" 2>&1 &
echo $! >"$PID_DIR/api.pid"
echo "[run] api server on http://$APP_HOST:$APP_PORT (log: $LOG_DIR/api.log)"

# ── Celery worker ────────────────────────────────────────
# Both queues in one worker: `gpu` runs training, `default` runs indexing.
#
# Pool choice matters. macOS starts subprocesses with `spawn`, which celery's
# prefork pool does not survive ("not enough values to unpack"), so it uses the
# thread pool there. On Linux prefork is kept, because a child exiting returns
# all of its GPU memory.
if [ "$(uname -s)" = "Darwin" ]; then
    CELERY_POOL="${CELERY_POOL:-threads}"
else
    CELERY_POOL="${CELERY_POOL:-prefork}"
fi

"$PYTHON" -m celery -A celery_app worker --loglevel=info --queues=gpu,default \
    --pool="$CELERY_POOL" --concurrency=2 \
    >"$LOG_DIR/worker.log" 2>&1 &
echo $! >"$PID_DIR/worker.pid"
echo "[run] celery worker started, $CELERY_POOL pool (log: $LOG_DIR/worker.log)"

echo "[run] ready. Press Ctrl-C to stop."
trap stop INT TERM
tail -f "$LOG_DIR/api.log" "$LOG_DIR/worker.log"
