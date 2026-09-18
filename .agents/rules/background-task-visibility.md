---
trigger: always_on
---

# Background Task Transparency & User Observability Directive

Whenever a long-running process or background task is executed:
1. **Immediate Task Specification:** State the human-readable purpose, exact command line, and task ID.
2. **Clickable Log Path:** Provide the full absolute path to the task's log file as a clickable markdown file link.
3. **Live Monitoring Command:** Always provide the user with the exact copy-paste shell command to tail or inspect the running process live in their terminal:
   ```bash
   tail -f "<log_file_path>"
   ```
4. **Process Inspection Command:** Provide a quick command to verify CPU/process health (e.g., `ps aux | grep <script_name>`).
5. **No Silent Waiting:** When reporting on a running background task, always summarize the most recent progress line from the log and the current runtime duration.
6. **Periodic Health Checks & Deadlock Detection:** Periodically inspect running background tasks (e.g. via schedule timers, log tailing, and CPU time inspection) to verify active progress. Immediately detect and alert the user if a process is deadlocked, stalled, or stuck in an infinite error loop.
