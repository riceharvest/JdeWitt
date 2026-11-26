# Phase 7 release matrix

Manifest: `verification/releases/matrix.manifest.json`
Pinned serve config: ctx 4096 / parallel 4 / cap 1024; registered suites removal 40, held_out 120, retention 241; registered caps removal 1024, held_out 1024, retention 1024; scale grid <= 2.0
complete: false   certified: 0 of 5 required lineages   certified_at_other_config: 0   UNMEASURED: 1   invalid: 0   failed: 4

## Lineages

| lineage | model_type | model | revision | quant | status | removal (shipped/wide) | held-out | maths | drift | fidelity min cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-4b | qwen3 | Qwen/Qwen3-4B-Instruct-2507 | cdbee75f17c0 | Q8_0 | failed(removal) | 5/38 / 3/38 | 30/112 | 158 -> 153 (p 0.0648) | 0.0606 in 0.02 | 0.9905 |
| qwen25-1.5b | qwen2 | Qwen/Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | Q8_0 | failed(removal) | 25/40 / 25/40 | 84/120 | 52 -> 11 (p 0.0001) | 0.44553 in 0.02 | 0.9909 |
| spark-1.7b | spark2_5 | Spark-X2.6-1.7B | 0e60633b345b19ba9f56ed1fda197acd271e8546 | Q4_K_M | failed(removal) | 12/15 / 9/15 | 18/40 | 68 -> 67 (p 1.0) | 0.0069 in 0.02 | 0.9687 |
| minicpm-v47 | minicpmv4_7 | openbmb/MiniCPM-V-4.7-1B | 286817f1132b998913a129a221c89b4402b4ab80 | Q4_K_M | failed(removal) | 38/38 / 38/38 | 103/103 | 154 -> 155 (p 1.0) | 0.0094 in 0.02 | 0.4149 |
| inkling-2-mini | inkling2_loop | Inkling-2-Mini | b85c3cfa615fc413cc9af3379f491b963f3b586a | IQ4_NL | UNMEASURED | - | - | - | - | - |
| spark-4b (spare) | spark2_5 | Spark-X2.6-4B | None | Q4_K_M | UNMEASURED | - | - | - | - | - |
| gemma4-31b (spare) | gemma4 | Gemma-4-31B q4_0 | None | Q4_0 | UNMEASURED | - | - | - | - | - |
| qwen38-27b (spare) | qwen3_5 | Qwen3.8-27B UD-Q4_K_M | None | Q4_K_M | UNMEASURED | - | - | - | - | - |

## LOSSES

The honest table: what each row lost against its own base, beside the reference side of the head-to-head where one exists.

    qwen3-4b  maths 158 -> 153    reference 159 -> 159    more loss than the reference (5 vs 0); paired p 0.0648 (not significant)
    qwen3-4b  code 22 -> 18    reference 22 -> 19    more loss than the reference (4 vs 3); paired p 0.5029 (not significant)
    qwen25-1.5b  maths 52 -> 11    reference 52 -> 45    more loss than the reference (41 vs 7); paired p 0.0001
    qwen25-1.5b  code 43 -> 0    reference 43 -> 39    more loss than the reference (43 vs 4); paired p 0.0001
    spark-1.7b  maths 68 -> 67    reference unmeasured    paired p 1.0 (not significant)
    spark-1.7b  code 44 -> 44    reference unmeasured    paired p 1.0 (not significant)
    minicpm-v47  maths 154 -> 155    reference unmeasured    paired p 1.0 (not significant)
    minicpm-v47  code 34 -> 38    reference unmeasured    paired p 0.4306 (not significant)

## Head-to-head

    qwen3-4b vs published build (p-e-w/Qwen3-4B-Instruct-2507-heretic q8_0): reference measured from waves
      removal: ours {'ctx': 4096, 'parallel': 4, 'cap': 1024} | reference {'ctx': 4096, 'parallel': 4, 'cap': 1024} -> same config
      held_out: ours {'ctx': 4096, 'parallel': 4, 'cap': 1024} | reference {'ctx': 4096, 'parallel': 4, 'cap': 1024} -> same config
      retention: ours {'ctx': 4096, 'parallel': 4, 'cap': 1024} | reference {'ctx': 4096, 'parallel': 4, 'cap': 1024} -> same config
    qwen25-1.5b vs published build (Isaak-Carter/Josiefied-Qwen2.5-1.5B-Instruct-abliterated-v1, Q8_0): reference measured from waves
      removal: ours {'ctx': 4096, 'parallel': 4, 'cap': 4096} | reference {'ctx': 4096, 'parallel': 4, 'cap': 4096} -> same config
      held_out: ours {'ctx': 4096, 'parallel': 4, 'cap': 4096} | reference {'ctx': 4096, 'parallel': 4, 'cap': 4096} -> same config
      retention: ours {'ctx': 4096, 'parallel': 4, 'cap': 1024} | reference {'ctx': 4096, 'parallel': 4, 'cap': 1024} -> same config

## Refusals

    (none)

## Next commands

    qwen3-4b (failed(removal)): uv run jdewitt release-certify --spec verification/p7-q3-4b/spec.json --out verification/p7-q3-4b/packet.json
    qwen25-1.5b (failed(removal)): uv run jdewitt release-certify --spec verification/p7-q25-15b/spec.json --out verification/p7-q25-15b/packet.json
    spark-1.7b (failed(removal)): uv run jdewitt release-certify --spec verification/releases/spark-1.7b/spec.json --out verification/releases/spark-1.7b/packet.json
    minicpm-v47 (failed(removal)): uv run jdewitt release-certify --spec verification/releases/minicpm-v47/spec.json --out verification/releases/minicpm-v47/packet.json
    inkling-2-mini (UNMEASURED): uv run jdewitt release-certify --spec verification/releases/inkling-2-mini/spec.json --out verification/releases/inkling-2-mini/packet.json
    spark-4b (UNMEASURED): uv run jdewitt release-certify --spec verification/releases/spark-4b/spec.json --out verification/releases/spark-4b/packet.json
    gemma4-31b (UNMEASURED): uv run jdewitt release-certify --spec verification/releases/gemma4-31b/spec.json --out verification/releases/gemma4-31b/packet.json
    qwen38-27b (UNMEASURED): uv run jdewitt release-certify --spec verification/releases/qwen38-27b/spec.json --out verification/releases/qwen38-27b/packet.json

The matrix is a count, not a claim: a row is `certified` only when every axis was measured at the pinned serve config and inside its registered floor. A row measured at another config is reported as `certified_at_other_config`, a row with no packet is `UNMEASURED`, and a row that narrowed its suite, lowered its cap or raised its scale past the grid is `invalid`, with a non-zero exit. A `failed(<axis>)` row is a measured result inside a registered band that it missed: it is reported as a failure and counted as one, never rounded into a pass.
