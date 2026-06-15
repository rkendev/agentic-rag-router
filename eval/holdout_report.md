# Held-out / red-team eval --- data/eval/holdout_questions.jsonl

Live router (claude-sonnet-4-6), 15 questions, generated 2026-06-15T13:19:26.684013+00:00. Same deterministic scoring as the frozen-set harness; this report is not a CI gate.

## Metrics

| metric | value |
| --- | --- |
| routing_accuracy | 1.0000 (11/11) |
| refusal_correctness | 1.0000 (4/4) |
| over_refusals | 0 |
| citation_coverage | 1.0000 (11/11) |

## Per-question outcomes

| id | label | probes | first_tool | refused | reason | cites |
| --- | --- | --- | --- | --- | --- | --- |
| HO01 | vector_only |  | vector_search | no |  | 5 |
| HO02 | vector_only |  | vector_search | no |  | 15 |
| HO03 | vector_only |  | vector_search | no |  | 20 |
| HO04 | sql_only |  | sql_query | no |  | 1 |
| HO05 | sql_only |  | sql_query | no |  | 1 |
| HO06 | sql_only |  | sql_query | no |  | 1 |
| HO07 | web_only |  | web_search | no |  | 10 |
| HO08 | web_only |  | web_search | no |  | 5 |
| HO09 | web_only |  | web_search | no |  | 5 |
| HO10 | no_answer |  | sql_query | yes | no_supporting_evidence | 0 |
| HO11 | no_answer |  | sql_query | yes | no_supporting_evidence | 0 |
| HO12 | no_answer |  | web_search | yes | no_supporting_evidence | 0 |
| HO13 | no_answer |  | vector_search | yes | no_supporting_evidence | 0 |
| HO14 | hybrid |  | sql_query | no |  | 6 |
| HO15 | hybrid |  | sql_query | no |  | 2 |
