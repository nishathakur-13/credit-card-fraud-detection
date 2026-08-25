---
name: fix-problem
description: 'Diagnose and fix software problems in a VS Code workspace. Use when debugging failures, fixing bugs, resolving errors, repairing broken behavior, or implementing a targeted code change.'
argument-hint: '[describe the problem, failing command, behavior, or file]'
user-invocable: true
---

# Fix a Problem

## Outcome

Resolve the reported problem with the smallest defensible change, while preserving unrelated user work and leaving behind executable evidence that the fix works.

## Procedure

1. **Establish the anchor.** Start from the most concrete available signal: a named file or symbol, failing test or command, diagnostic, stack trace, or reproducible behavior. Search narrowly and read the owning implementation, one nearby caller or test, and relevant configuration only as needed.
2. **State a local hypothesis.** Before editing, identify the code path that directly decides the behavior and write one falsifiable explanation for the failure. Name one cheap check that could disconfirm it. If the request is underspecified, infer the expected behavior from tests, call sites, documentation, and existing conventions; ask only when those sources conflict.
3. **Choose the smallest probe or fix.** Prefer a reversible, focused edit that tests the hypothesis and addresses the root cause. Preserve public APIs and local style. Do not refactor unrelated code, overwrite user changes, or add complexity without a clear need.
4. **Validate immediately.** After the first substantive edit, run the narrowest executable check available in this order: the failing behavior or test, a focused test for the touched slice, or a narrow compile, typecheck, or lint command. Do not resume broad exploration before this check.
5. **Interpret the result.**
   - If validation fails but supports the hypothesis, repair the same slice and rerun the same check.
   - If validation disproves the hypothesis, take one nearby hop to the code that more directly controls the behavior, revise the hypothesis, and make the smallest next edit.
   - If validation is ambiguous, perform one nearby read of a related test or call site, then choose between local repair and the one-hop investigation.
   - Stop after three focused repair attempts in the same file and report the blocker with the evidence collected.
6. **Broaden only when justified.** Once the focused check passes, run relevant integration, regression, or full-project checks when the change crosses module boundaries or affects shared behavior. Keep the scope proportional to risk.
7. **Close the loop.** Report the root cause, files changed, checks run and their outcomes, and any remaining test gap or assumption. Mention unrelated pre-existing failures without attempting to fix them.

## Quality Criteria

- The behavior is reproduced or the failure signal is understood before editing.
- The hypothesis and disconfirming check are explicit.
- The change is minimal, root-cause-oriented, and consistent with the repository.
- A post-edit executable validation was run whenever the environment provided one.
- No unrelated user changes were reverted.
- The final report distinguishes verified results from remaining uncertainty.
