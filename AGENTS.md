# AGENTS.md

This repository's agents must follow these rules.

## 1. Permission-first workflow

When asked to do something:

- First search the codebase for possible existing implementations.
- Do not immediately edit or create code.
- Discuss with the user what should be done, including options and tradeoffs.
- Seek explicit permission before proceeding.
- Only implement after permission is granted.

## 2. Python annotation and comment syntax

Follow the workplace coding syntax:

- Use Python type annotations.
- After each `def ...:` signature, write a short comment describing the logic and API of the code.

Example:

```python
def compute_score(x: Tensor, y: Tensor) -> Tensor:
    '''
    <Here describe the logic of the code (no need to add Logic: xxx)>
    Args:
      <Here describe the inputs>
    Returns:
      <Here describe the returns>
    Raises:
      <If raises any error > 
    '''  
    ...
```


- In the comment describe the shape of the tensor, or other requirement (Hermitian, Unitary, anti-Hermitian, etc.).

- Inside the function, use comment to explain hacking steps. (such as, bit-wise masking, strange steps for max performance) Although, do not use comment on easy steps. Python code should already clear enough

- If type annotation is helpful in terms of understanding the structure of the code (such as, nested list, tuple, etc), then use annotation

- When defining a tensor, write arguments once a line: such as
```python
t = torch.tensor(
  data,
  device = torch.device('cuda'),
  dtype = torch.complex128)
'''

- Use latex format for equations. Use r-string if latex is included. 

- Keep the logic in one line 

## 3. Always use CUDA for matrix computation

Always use CUDA for matrix computation.

- Move matrices/tensors to CUDA before matrix operations.
- Use CUDA-backed matrix operations.
- Do not use CPU matrix computation unless explicitly permitted.

## 4. Do not overdesign

Do not overdesign unless specified.

- Keep the code as simple as possible to pass the test.
- Add abstractions, configuration, or extensibility only when additional requirements are planned.

## 5. Inline one-off logic

For variables or functions that are only used once or twice, inline the related logic.

- Avoid unnecessary helper functions or intermediate variables for single-use or double-use cases.
- Prefer direct, local logic unless reuse or clarity justifies extraction.