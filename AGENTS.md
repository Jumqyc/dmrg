# AGENTS.md

This repository's agents must follow these rules.  The shared rules — permission-first,
how to talk, what not to do, verification, report — live in `~/.dsh/AGENTS.md` and are
not repeated here: what follows is only what this repository adds to them or states
differently, so both files together are the whole instruction set.

## 4. Code

Python here; most of it is language-agnostic. Follow the workplace coding syntax.

- No long explanatory preamble at the top of a file: a module docstring says what the
  module is and the API it exposes, not the reasoning or the history behind it.
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
- Never swallow an exception: no bare `except`, no silent fallback on failure.
- No debug prints, `breakpoint()`, dead code or commented-out blocks in a finished change.

## 7. Notebooks

- A markdown cell first — what this does and why — then one code cell that does it.
- Keep the scope minimal: imports, then one or two adjacent cells for the task.
- Plot, or format a compact table, instead of printing walls of text.
- Every word describes the current code. No narration of earlier attempts, no "as we saw".
- Restart-and-run-all must work top to bottom: no hidden state, no out-of-order cells.
- Clear stored outputs and execution counts before committing, unless the user wants them kept.
