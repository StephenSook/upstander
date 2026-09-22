#!/bin/sh
# Stop EVERY Upstander server (uvicorn waits on open websockets, so a plain kill can leave copies
# running, each with its own state), then start exactly one.
cd "$(dirname "$0")/.." || exit 1
pkill -9 -f "uvicorn app.main:app" 2>/dev/null
sleep 1
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 2 > server.log 2>&1 &
sleep 3
echo "servers running: $(pgrep -f 'uvicorn app.main:app' | wc -l | tr -d ' ') (must be 1)"
