#!/usr/bin/env bash
# Usage:
#   ./start.sh          — start API + frontend
#   ./start.sh stop     — stop both

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$REPO/.app.pids"

# ── load app.env ──────────────────────────────────────────────────────────────
if [[ -f "$REPO/app.env" ]]; then
  while IFS='=' read -r key value; do
    # skip comments and blank lines
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    # resolve relative paths against REPO
    if [[ "$key" == *PATH* && "$value" != /* ]]; then
      value="$REPO/$value"
    fi
    export "$key=$value"
  done < "$REPO/app.env"
fi

API_DIR="$REPO/src_new/frontend/api"
WEB_DIR="$REPO/src_new/frontend/webapp"
PYTHON="$REPO/myenv/bin/python"
UVICORN="$REPO/myenv/bin/uvicorn"

# ── stop ──────────────────────────────────────────────────────────────────────
stop() {
  if [[ ! -f "$PIDFILE" ]]; then
    echo "No running instance found."
    return
  fi
  echo "Stopping..."
  while IFS= read -r pid; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "  killed PID $pid"
    fi
  done < "$PIDFILE"
  rm -f "$PIDFILE"
  echo "Done."
}

if [[ "${1:-}" == "stop" ]]; then
  stop
  exit 0
fi

# ── guard: already running? ───────────────────────────────────────────────────
if [[ -f "$PIDFILE" ]]; then
  echo "Already running (delete $PIDFILE to force restart)."
  exit 1
fi

# ── start API ─────────────────────────────────────────────────────────────────
echo "Starting API server..."
cd "$API_DIR"
PYTHONPATH="$API_DIR" "$UVICORN" main:app --host 127.0.0.1 --port 8000 \
  > "$REPO/api.log" 2>&1 &
API_PID=$!

# ── start frontend ────────────────────────────────────────────────────────────
echo "Starting frontend..."
cd "$WEB_DIR"
npm run dev > "$REPO/frontend.log" 2>&1 &
WEB_PID=$!

# ── save pids ─────────────────────────────────────────────────────────────────
printf '%s\n%s\n' "$API_PID" "$WEB_PID" > "$PIDFILE"

# ── wait for both to be ready ─────────────────────────────────────────────────
echo -n "Waiting for API  "
for i in $(seq 1 20); do
  sleep 1
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo " ready"
    break
  fi
  echo -n "."
done

echo -n "Waiting for UI   "
for i in $(seq 1 20); do
  sleep 1
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null | grep -q "200"; then
    echo " ready"
    break
  fi
  echo -n "."
done

echo ""
echo "  API  →  http://localhost:8000"
echo "  App  →  http://localhost:3000"
echo ""
echo "Run './start.sh stop' to shut down."
