#!/usr/bin/env bash
# Wait for a running training PID, then run the rest of the pipeline.
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
exec "$(dirname "$0")/run_all.sh"
