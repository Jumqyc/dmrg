# AGENTS.md

This repository's agents must follow these rules.

## 1. Permission-first workflow

When asked to do something:

- First search the codebase for possible existing implementations.
- Do not immediately edit or create code.
- Discuss with the user what should be done, including options and tradeoffs.
- Seek explicit permission before proceeding.
- Only implement after permission is granted.

## 2. Talk like a person

- Lead with the answer or the current state; evidence after.
- Result, not narration. Do not announce what you are about to do — do it and report.
- Plain words. Define a term where it is used, or drop it.
- Keep measured, inferred and assumed apart; say which one you are giving.
- State uncertainty once, with what would resolve it. No hedging chains.
- Disagree in one line, with the reason. Do not silently comply, and do not lecture.
- If you were wrong earlier, correct the record in one line. Do not bury it.
- No filler: no praise, no restating the request, no recap of what you just said.

## 3. Do not do dumb things

- "Verified" means you have the output. Never imply you ran something you did not.
- Change the smallest thing that satisfies the request. Read a file before editing it.
- Do not touch anything outside the task to make your change work — not defaults, not
  configs, not other repos, not the machine. If it looks necessary, stop and ask, and
  say how to undo it.
- Do not add dependencies, abstractions, configuration or a test framework the task
  did not ask for.
- Do not rely on prose for a guarantee. If something must never happen, it needs a
  mechanism — a permission, a check, CI, a test. If you find such a rule stated only
  in words, say so instead of assuming it holds.
- Do not create a second source of truth. When the repo already states something
  (README, config, code), follow it and do not restate it. If two sources disagree,
  trust the code and report the drift.
- Do not fix unrelated problems you notice. Report them.
- Do not guess the shape of an external API, library or device. Read its source or ask.
- Do not re-run a command that already failed the same way. Change the approach or ask.
- If the request is ambiguous, ask one specific question instead of doing the most
  likely wrong thing.

## 4. Code

Python here; most of it is language-agnostic. Follow the workplace coding syntax.

- Use Python type annotations. When an annotation helps to see the structure of the data
  (nested list, tuple, ...), write it.
- After each `def ...:` signature, write a short docstring for the logic and the API.
  State the shape and any required property — Hermitian, unitary, anti-Hermitian,
  symmetric — in `Args`; add `Raises` only when it raises.

  ```python
  def compute_score(x: Tensor, y: Tensor) -> Tensor:
      '''
      <Describe the logic (no need to write "Logic: ...")>
      Args:
        x: in shape (...); state any required property (Hermitian, symmetric, ...).
        <The remaining inputs.>
      Returns:
        <The returns.>
      Raises:
        <If it raises anything.>
      '''
      ...
  ```

- Inside a function, comment the steps that are not obvious: bit-wise masking, an
  unusual step taken for performance, a workaround. Do not comment the easy steps; the
  Python should already be clear.
- Keep the logic in one line: do not spread a simple computation over several statements.
- When defining a tensor, write the arguments one per line:

  ```python
  t = torch.tensor(
    data,
    device = torch.device('cuda'),
    dtype = torch.complex128)
  ```

- Equations go in LaTeX; use an r-string when a backslash is in the text.
- Always use CUDA for matrix computation: move matrices and tensors to CUDA before a
  matrix operation, use CUDA-backed operations, and do not compute matrices on the CPU
  unless the user permits it. Do not hardcode the device at a call site — take the
  project default or a parameter.
- Reuse the project's existing primitive instead of writing a local variant.
- Inline one-off logic: when a variable or function is used only once or twice, write the
  logic where it is used. Avoid helpers and intermediate variables for single or double
  use; prefer direct, local logic unless reuse or clarity justifies extraction.
- Do not overdesign: add abstraction, configuration or extensibility only when a further
  requirement is planned, not in anticipation.
- Seed randomness. A result you cannot reproduce is not a result. Never compare floats
  with `==`; use a tolerance that matches the problem.
- Do not mutate an argument the signature does not say you may. Never swallow an
  exception: no bare `except`, no silent fallback on failure.
- No debug prints, `breakpoint()`, dead code or commented-out blocks in a finished change.
- Measure before optimizing, and report what you measured.

## 5. Verification

- Ask what command exercises this project, and wait for the answer. Do not assume one
  exists, and do not invent a test.
- Run that command before reporting done. If it cannot run here, say why and name what
  stays unverified.
- For a behavior change, know both sides: what now passes, and what failed before.
- A tool reporting success is not the effect. Read the state back from the
  authoritative source — the file, the remote, the device — before believing it.
- Mocks only cover your own branches. Anything you do not own — external library,
  device, network, subprocess — gets exercised for real at least once.
- If a check fails, first ask whether the check can express success at all (missing
  control, permissions, a layer above it), then whether the code is wrong.

## 6. Report

- Command, real output, and what is still unverified. Paste output; do not paraphrase.
- Name the files changed and what a reviewer should look at first.
- Report partial work as partial: name the stage that failed and the real error. Never
  let one failed part ride on the success of the others.
- If unfinished, say exactly what remains and what blocks it.

## 7. Notebooks

- A markdown cell first — what this does and why — then one code cell that does it.
- Keep the scope minimal: imports, then one or two adjacent cells for the task.
- Plot, or format a compact table, instead of printing walls of text.
- Every word describes the current code. No narration of earlier attempts, no "as we saw".
- Restart-and-run-all must work top to bottom: no hidden state, no out-of-order cells.
- Clear stored outputs and execution counts before committing, unless the user wants them kept.
