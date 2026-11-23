---
name: task-commit
description: Safely inspects working tree changes, checks for sensitive files, drafts a conventional commit message, presents a pre-commit report for user review, and executes the git commit upon approval. Handles both standard task commits and automated release workflows with Git tagging. Use whenever asked to commit changes, save progress to git, or create a release commit.
---

# Task Commit Skill

When triggered, act as a strict Git Commit and Release Manager. Follow this exact 5-step sequence:

---

## Step 1: Inspect (Silent)

1. **Detect Branch & Release Intent**:
   - Check the active git branch (`git branch --show-current` or `pycharm` MCP `git_status`).
   - Check if the user's prompt indicates a release workflow (e.g., *"commit for a new release"*, *"release commit"*, *"cut a release"*, *"commit v1.0.0"*, or `--release`).
   - If **Release Mode** is detected, verify tests pass in `D:\anaconda3\envs\ml_transdat`.

2. **Check Working Tree Status**:
   - Run `git status -u` (or `git status --porcelain`). *(Note: On Windows sandbox environments where `.git` is isolated, run git inspection and commit commands with `BypassSandbox: true`)*.
   - If the working tree is completely clean, inform the user immediately and stop execution.
   - **Branch Safety Check**: If on `main` and staging a brand-new experimental unit (`experiments/<slug>/`), flag this in the pre-commit report so the user can confirm whether they want it on `main` or an `exp/<slug>` branch.

3. **Check Diffs & Existing Stage**:
   - Run `git diff` and `git diff --cached` (or `git diff --staged`).

4. **Sensitive File & Cache Scan**:
   - Never propose staging `.env*`, credentials/keys (`*.pem`, `*.key`, `id_rsa*`), model binaries (`*.pkl`, `*.pt`, `*.pth`, `experiments/*/models/`), or caches/artifacts (`__pycache__`, `*.pyc`, `.pytest_cache/`, `*.log`, `Thumbs.db`, `.DS_Store`).

---

## Step 2: Generate the Pre-Commit Report

Do **not** stage (`git add`) or commit anything yet. Present a clear, structured report:

1. **Branch & Mode**: `<current-branch>` — Standard Task Commit OR **Release Commit (`v<version>`)**
2. **Files to Stage**: Exact file paths with a 1-sentence summary for each.
3. **Excluded / Ignored Files** *(omit if none)*: Untracked or cache files left unstaged.
4. **Proposed Commit Message**: Conventional Commits (`<type>(<optional-scope>): <concise description>`, types: `feat`, `fix`, `refactor`, `test`, `docs`, `perf`, `chore`; or `chore(release): v<version>`).
5. **Git Tag to Create** *(Release Mode only)*: `v<version>`.

---

## Step 3: Pause for Review (Interactive Buttons)

Call `ask_question` before staging or committing:

### A. For Standard Commits:
- **Question**: `"Review the pre-commit report above. How would you like to proceed?"`
- **Options**:
  1. `"(Recommended) Approve and execute the commit as proposed"`
  2. `"Edit the commit message before committing"`
  3. `"Cancel the commit"`
- **is_multi_select**: `false`

### B. For Release Commits:
- **Question**: `"Review the release pre-commit report above. How would you like to proceed?"`
- **Options**:
  1. `"(Recommended) Approve release commit and create Git tag v<version>"`
  2. `"Approve release commit only (do not create Git tag)"`
  3. `"Edit the commit message before committing"`
  4. `"Cancel the release commit"`
- **is_multi_select**: `false`

Wait for explicit user approval via `ask_question` before proceeding to Step 4. **Never proceed to Step 4 in the same turn.**

---

## Step 4: Execute (Upon Explicit Approval)

1. **Targeted Staging**: Run `git add <file1> <file2> ...` targeting **only** the approved files (never `git add .` or `git add -A`).
2. **Safe Commit Execution (PowerShell-Safe)**: Use multiple `-m` arguments (`git commit -m "<subject>" -m "<body>"`) or `git commit -F <temp_msg_file>`.
3. **Release Git Tagging (Release Mode only, upon user approval)**:
   ```powershell
   git tag -a v<version> -m "ML TransDat v<version>"
   git tag -l -n1 v<version>
   ```
4. **Pushing Policy**: **NEVER** run `git push`. Leave remote pushing strictly to the user.

---

## Step 5: Post-Commit Verification

1. Run `git log -1 --stat` and `git status`.
2. If in Release Mode and tagged, run `git tag -l -n1 v<version>`.
3. Present confirmation with commit hash, subject, any remaining unstaged files, and (for Release Mode) `git push origin main --tags`.
