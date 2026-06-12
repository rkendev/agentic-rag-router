#!/usr/bin/env bash
# Stop hook — emits the FF-merge + push + Windows-mirror-pull commands
# when a session ends on a non-`main` branch. Per
# feedback_task_close_sequence.md — saves user from asking after every task.
#
# Council plan 2026-05-01 §7 item 1.
#
# NOTE: Windows mirror path prefix below is Roy's local convention.
# Forkers on different OS or different mirror layout: edit the
# 'C:\Users\...' path below or replace with $WINDOWS_MIRROR_BASE env var.
# Hook is non-fatal — wrong path just produces wrong copy-paste output.
set -euo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
PROJECT="$(basename "$(pwd)")"

# No close-sequence needed if we're on main (no merge to do) or detached.
if [[ "$BRANCH" == "main" ]] || [[ "$BRANCH" == "unknown" ]] || [[ "$BRANCH" == "HEAD" ]]; then
    exit 0
fi

cat <<EOF

╔═══════════════════════════════════════════════════════════════╗
║  TASK CLOSE SEQUENCE  (auto-emitted by Stop hook)             ║
╚═══════════════════════════════════════════════════════════════╝

# On VPS:
cd /root/projects/AI-Engineering/$PROJECT
git log --oneline -3
git checkout main
git merge --ff-only $BRANCH
git push origin main
git branch -d $BRANCH

# On Windows:
cd "C:\\Users\\User\\Downloads\\Architecting_Domain_Driven\\Claude-Cowork\\Projects\\$PROJECT-from-git"
git pull origin main
git log --oneline -3

EOF
