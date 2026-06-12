---
description: Emit the task close-sequence (FF-merge + push + Windows mirror pull) for the current branch. On-demand version of the Stop hook.
---

Detect the current branch and project, then emit the standard close-sequence per `feedback_task_close_sequence.md`.

## Steps

1. Run `git rev-parse --abbrev-ref HEAD` to get the current branch.
2. Use `basename "$(pwd)"` to get the project name.
3. If branch is `main` or detached, output: "Already on main — no close-sequence needed." and stop.
4. Otherwise output two code blocks for the user to copy:

   Block 1 (bash, VPS-side):
   ```bash
   cd /root/projects/AI-Engineering/<project>
   git log --oneline -3
   git checkout main
   git merge --ff-only <branch>
   git push origin main
   git branch -d <branch>
   ```

   Block 2 (PowerShell, Windows-side):
   ```powershell
   cd "C:\Users\User\Downloads\Architecting_Domain_Driven\Claude-Cowork\Projects\<project>-from-git"
   git pull origin main
   git log --oneline -3
   ```

Substitute `<project>` and `<branch>` with the values from steps 1-2. Do NOT execute the commands — just emit them for copy-paste. The user runs them themselves to retain explicit control over the merge.
