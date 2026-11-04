# Document directory organization Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep the project root readable by grouping current documents and retaining reproducibility.
**Architecture:** Reports go to `reports/`, notebooks to `notebooks/`, supporting material to `docs/`, and validation records to `validation/`. Every producer and consumer uses those paths.
**Tech Stack:** Python, Markdown, XeLaTeX, Jupyter, PowerShell, Git.
**Spec:** User-approved four-folder organization and commit/push request, 2026-11-05.

## Global Constraints

- Preserve active training and all models, predictions, histories, and scientific figures.
- Keep only the current document files; Git history retains earlier revisions.
- Maintain links and document regeneration; commit and push after verification.

## Review Focus

- Report and gallery image paths resolve relative to `reports/`.
- The notebook works when launched in `notebooks/` and defaults to no training.
- Generated validation records and ZIP manifests use the new paths.
- Public results remain the existing snapshot; relocation does not imply new experiment completion.
- Live computation files must not be overwritten by the public snapshot.

## Tasks

- [x] Move documents and rebase existing links without touching numerical results.
- [x] Update report, notebook, gallery and finalization scripts plus their existing integration checks.
- [x] Verify moved links, execute notebook, verify LaTeX compilation and numerical-file preservation.
- [x] Apply the same bounded migration to the live folder and review.

Finish: commit and push the verified migration.

Implementation note: the public Git checkout is separate from the active computation directory. No extra checkout is needed for isolation. The user explicitly requested implementation and push, so no additional approval gate is needed.

Verification: 39 targeted checks passed; 15 notebook code cells executed without errors; 347 GitHub formulas preserved; three delivered PDFs byte-identical; relocated TeX compiled with 0 warnings and identical text on all 56 report pages. Independent review found no actionable issues.
