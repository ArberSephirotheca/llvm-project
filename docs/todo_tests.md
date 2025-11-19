# Test ideas for CPS interpreter masks

- Multi-lane branch reconvergence: two lanes, one true/one false, verify parent resumes per lane and merge entry pops when expectedMask shrinks.
- Collective wait/shrink: child block with a collective expecting all active lanes; resolve one lane elsewhere, shrink expected, and ensure the collective resumes remaining lanes.
- Nested if: lanes split outer/inner, ensure inner/outer merge entries pop independently and parent continuations per lane run.
