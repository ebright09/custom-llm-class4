# TrashGPT verification

Verified September 29, 2026 (America/Los_Angeles).

## Automated checks

`python -m unittest test_trashgpt test_language_evals test_corpus`: **19 tests passed**.

- All four before/after result sets contain all 48 distinct cases; totals match raw case records.
- Four saved models, each sampled at three temperatures, match direct inference exactly.
- The expanded model reproduces `a goose .` at temperature 0.8 and seed 2029.
- Model fingerprints stay unchanged after generation and match original evaluations.
- Empty replies, unknown vocabulary, context truncation, and invalid input are handled.
- All approved evidence routes load; arbitrary files and nonlocal origins/hosts are rejected.
- Original pinned helper and suite hashes are unchanged.
- JavaScript syntax, launcher syntax, and relative documentation links checked.

## Actual browser checks

Used the local Codex browser at desktop sizes 1280×720 and 1440×1000, and a narrow 390×844 viewport.

- All five learning stops render. Token ID changes correctly from 28 to 105 between experiments.
- Before/after embedding controls, saved training milestones, recorded attention, and evaluation stage selection work.
- All 48 evaluation tiles appear; case 28 shows actual trained and untrained predictions.
- Expanded trained replies: goose, clock, and opposite examples match the saved evidence.
- Starter trained unknown-word prompt shows `[empty response]` and names the unknown words.
- Starter untrained at temperature 1.2 handles a long prompt and reports truncation.
- Keyboard Enter activates an embedding control; focus remains on it after rerender; Tab reaches the next control.
- Narrow layout and expanded results tables have no horizontal overflow.
- Browser console reports no errors or warnings during these checks.
- Transcript download produces actual JSON with three interactions, settings, run identities, and model hashes.

## Saved browser evidence

- `browser-transcript.json`: actual downloaded three-turn browser transcript, copied without editing.
- `trashgpt-overview.png`: actual desktop browser viewport.
- `goose-interaction.png`: actual browser reply and separate authored commentary, alongside the lesson.

Original notebooks, models, fixed test suite, training helpers, and saved experiment artifacts were not modified. No new training or public deployment was performed.
