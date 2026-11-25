#!/usr/bin/env bash
# User-level guard for the shared box (no root: no earlyoom, no hardware watchdog). Run it in tmux `lawrag-watchdog`.
# Every 20 s: log memory / disk / GPU once a minute (evidence for the next crash); if available RAM < 6 GB or free disk < 8 GB,
# kill ONE of our own tmux sessions (newest lawrag-run-* first, then lawrag-llm-*), never anything else.
# The box hard-hung on 2026-11-22 (no OOM message, no swap) and the disk filled on 2026-11-25; the launch guard only looks once, at launch.
cd "$(dirname "$0")/.." || exit 1
LOG=data/runs/logs/watchdog.log
mkdir -p data/runs/logs
i=0
while true; do
  ram=$(free -g | awk '/^Mem:/{print $7}')
  disk=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ $((i % 3)) -eq 0 ]; then
    echo "$(date '+%F %T') ram_avail=${ram}G disk_free=${disk}G gpu=$(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | tr '\n' ';')" >> "$LOG"
  fi
  if [ "$ram" -lt 6 ] || [ "$disk" -lt 8 ]; then
    victim=$(tmux ls -F '#{session_created} #{session_name}' 2>/dev/null | grep -E ' lawrag-run-' | sort -rn | head -1 | cut -d' ' -f2)
    [ -z "$victim" ] && victim=$(tmux ls -F '#{session_created} #{session_name}' 2>/dev/null | grep -E ' lawrag-llm-' | sort -rn | head -1 | cut -d' ' -f2)
    if [ -n "$victim" ]; then
      echo "$(date '+%F %T') LOW RESOURCES ram=${ram}G disk=${disk}G -> killing $victim" >> "$LOG"
      tmux kill-session -t "$victim"
    fi
  fi
  i=$((i + 1))
  sleep 20
done
