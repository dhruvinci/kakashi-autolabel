#!/usr/bin/env bash
# Restart the autolabel orchestrator if it is not running (state is file-based, so restarts are safe).
# Installed in Dhruva's crontab 2026-10-05 (every 10 min + @reboot). Disable: touch autolabel/v1/orchestrator.stop
# and remove the crontab line.
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd "$(dirname "$0")" || exit 1
[ -f ../autolabel/v1/orchestrator.stop ] && exit 0
if ! pgrep -f '^\.\./\.venv/bin/python -u autolabel_orchestrate\.py' >/dev/null; then
    echo "$(date '+%F %T') watchdog: starting orchestrator" >> ../autolabel/v1/watchdog.log
    # Block system sleep (incl. lid-close suspend) while the pipeline runs: a 9 h suspend on 2026-10-09 froze all work.
    nohup systemd-inhibit --what=sleep --mode=block --who=autolabel \
        --why="Kakashi autolabel pipeline running (touch autolabel/v1/orchestrator.stop to release)" \
        ../.venv/bin/python -u autolabel_orchestrate.py >> ../autolabel/v1/orchestrator.out 2>&1 &
fi
