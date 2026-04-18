# Project Instructions

## Stack
Python research project focused on ML / RL. Follow existing patterns in the codebase before introducing new ones.

## Core Rules
- NEVER delete files without explicit permission — always ask first
- Do not add comments to code unless asked
- Keep changes minimal and targeted — avoid refactoring unrelated code
- Preserve reproducibility: do not silently change random seeds, hyperparameters, or experiment configs

## Planning
For any change touching more than one file, present a plan and wait for approval before implementing.
Reference: see docs/architecture.md if it exists.

## Python Standards
- Use type hints on all function signatures
- Prefer explicit over implicit — no magic imports or hidden side effects
- Keep functions small and single-purpose
- Use `if __name__ == "__main__":` guards in scripts

## ML / RL Specifics
- Hyperparameters go in config files or argparse — never hardcoded in logic
- Random seeds must be set explicitly and documented
- Log training runs with enough detail to reproduce results
- Do not modify evaluation logic without flagging it — evaluation changes affect result validity

## Dependencies
- Do not add new packages without asking
- Pin versions when adding to requirements.txt or pyproject.toml

## Key Files
- Experiment configs: see configs/
- Environment definitions: see envs/
- Training entry points: see scripts/ or train.py

## Testing
Run existing tests after any non-trivial change:
```
pytest
```
If no tests exist for a module, note it but do not generate tests unprompted.
