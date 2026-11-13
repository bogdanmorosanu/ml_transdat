---
name: code-audit
description: Audits code quality, typing, and formatting across Python scripts, utilities, and ML pipelines using Ruff and type checks. Use before finishing significant development sessions.
---

# Code Audit Skill (`code-audit`)

## Purpose
This skill executes static analysis, formatting checks, and code hygiene audits across the `ml_transdat` repository. It verifies that Python formatting, type conventions, explicit error handling, and file paths comply with project invariants.

---

## Audit Workflow

### 1. Python Code Hygiene & Ruff Lint
Run the Ruff linter and formatter check from the designated environment:
```powershell
D:\anaconda3\envs\ml_transdat\python.exe -m ruff check .
D:\anaconda3\envs\ml_transdat\python.exe -m ruff format --check .
```
*(If Ruff is not installed in the environment, skip external linting and inspect manually).*

### 2. File Path & Portability Verification
Scan modified files to ensure:
- No machine-specific absolute file paths (e.g., `file:///D:/...` or hardcoded Windows drive letters in committed scripts/docs).
- All file paths and links use workspace-relative paths.

### 3. Machine Learning & Invariant Safeguards
- Verify that ground-truth grid files in `pytransdat/grids/` have not been altered.
- Ensure that experimental scripts cleanly catch domain exceptions and avoid bare `except:` blocks.
