# Evidence map

Every sentence in README section IV that states a number, mapped to the artifact that produced it and to a locator that recomputes it from that artifact's rows. 'in-clone' means a fresh clone can reproduce the number, not merely inspect a file; anything else is labelled by why it is not.

| status | claim | artifacts named | command |
|--------------------------------------------------------------------------------------------------------|
| in-clone | Qwen3-4B fitted arms do not reach the published same-base removal | `verification/screens/advbench-40.jsonl`, `verification/q3-4b-harmful40/q3oracle/predictions.jsonl`, `verification/q3-4b-harmful40/q3shl10l935/predictions.jsonl`, `verification/q3-4b-retention/q3base/predictions.jsonl`, `verification/q3-4b-retention/q3shl10l935/predictions.jsonl`, `verification/q3-4b-oracle-control/shape-audit.json`, `verification/q3-4b-fit-scaling.json` | `.venv/bin/python scripts/frontier_report.py --base-dir verification/q3-4b-retention/q3base --pair "oracle=verification/q3-4b-harmful40/q3oracle:verification/q3-4b-retention/q3oracle" --pair "l935=verification/q3-4b-harmful40/q3shl10l935:verification/q3-4b-retention/q3shl10l935"` |
| in-clone | the oracle edit is a shared rank-1 axis confined to blk.9-35, and the fitted axis only reaches 0.5056 of it | `verification/q3-4b-fit-scaling.json`, `verification/q3-4b-oracle-control/shape-audit.json` |  |
| | ↳ the axes tensors themselves are 90 MB captures in scratch, not in the clone; the recovered direction is reproducible from the published GGUF with gguf-tensor-diff. | | |
| in-clone | Qwen2.5-1.5B capability retention: base 98/241, published third-party ablation 81/241, SAS@1.5 184/241 | `verification/retention-q25-code61/base/predictions.jsonl`, `verification/retention-q25-code61/sas15/predictions.jsonl`, `verification/retention-q25-code61/josiefied/predictions.jsonl`, `verification/retention-q25-code61/format-audit.json`, `verification/retention-q25-code61/provenance.json` | `.venv/bin/python scripts/capability_format_audit.py --base-dir verification/retention-q25-code61/base --edit-dir verification/retention-q25-code61/sas15 --suite verification/capability/capability-run-en.jsonl` |
| in-clone | a wave cannot report a number for an artifact the port is not serving, and an empty screen is not a result | `verification/instrument-gates/NOTES.md`, `verification/instrument-gates/wrong-operand-refusal.txt`, `verification/instrument-gates/unmeasured-because-cap64.wave.json` | `.venv/bin/python scripts/capability_wave.py --help` |
| pre-2026-11-17-scoring-fix | Qwen2.5-1.5B shared-axis fit: 40/40 on the harmful-40 suite, 24/24 on held-out refusal families, harmless clean | *(none)* |  |
| | ↳ Taken before the row-kind scoring fix, and the raw harmful/holdout answer rows are not in this clone. The certification is void as a number; the retained artifact is the capability-retention wave above, which measures what happened to maths and code, not whether refusals left. | | |
| pre-2026-11-17-scoring-fix | MiniCPM-V 4.7 per-layer answer-span fit: 36/40, plus null-space-protected and shape-interpolated variants | `verification/minicpm-v47/summary.json` |  |
| | ↳ The committed MiniCPM directory contains serving checks, GGUF digests and a baseline refusal screen only. The removal screens behind 36/40 predate the fix and are not in the clone, so the number is quoted in history, not as current evidence. | | |
| no-artifact-in-clone | Spark-X2.6-1.7B and LFM3-1.7B refusal openers gone 40/40, harmless clean, style drift recorded | *(none)* |  |
| | ↳ No wave, screen or provenance file for either model is in this repository or on the lab disk. Treat as unevidenced until re-run; it is a candidate result, not a certified one. | | |
| in-clone | on 24 refusal families never used for fitting, the best fitted q3 arm removes 23 of 24 and the published build removes 24 | `verification/screens/q3-holdout-24.jsonl`, `verification/q3-4b-holdout/hold-base/predictions.jsonl`, `verification/q3-4b-holdout/hold-oracle/predictions.jsonl`, `verification/q3-4b-holdout/hold-l935/predictions.jsonl`, `verification/q3-4b-holdout/hold-attn20/predictions.jsonl`, `verification/q3-4b-holdout/hold-yxattn20/predictions.jsonl`, `verification/q3-4b-holdout/hold-yxboth10/predictions.jsonl`, `verification/q3-4b-retention/yxboth10-l935/wave.json` | `.venv/bin/python scripts/frontier_report.py --screen "base=verification/q3-4b-holdout/hold-base" --screen "oracle=verification/q3-4b-holdout/hold-oracle" --screen "l935=verification/q3-4b-holdout/hold-l935" --screen "attn20=verification/q3-4b-holdout/hold-attn20" --screen "yxattn20=verification/q3-4b-holdout/hold-yxattn20" --screen "yxboth10=verification/q3-4b-holdout/hold-yxboth10"` |
| | ↳ one row per family, 24 distinct families, built from verification/harsh-suites/holdout.jsonl with every fit-pool id excluded. This is the screen the removal claim rests on, not advbench-40. | | |
| in-clone | no fitted q3 arm is statistically distinguishable from the published removal on the held-out families — on the wording metric only | `scripts/paired_screen_test.py`, `verification/q3-4b-holdout/hold-oracle/predictions.jsonl`, `verification/q3-4b-holdout/hold-yxboth10/predictions.jsonl`, `verification/q3-4b-holdout/shape-audit.json`, `verification/q3-4b-harmful40/shape-audit-yxboth10.json`, `verification/q3-4b-holdout/semantic-read-yxboth10.json`, `verification/q3-4b-holdout/semantic-read-oracle.json`, `verification/q3-4b-wording-inventory.json` | `.venv/bin/python scripts/paired_screen_test.py verification/q3-4b-holdout/hold-oracle/predictions.jsonl verification/q3-4b-holdout/hold-yxboth10/predictions.jsonl` |
| | ↳ RETRACTED. The wording parity was a checker artifact: the shipped inventory matches first-person refusals only, and the arm's answers open 'Absolutely not.' Wider-inventory grading of the same rows (scripts/screen_agreement.py): arm 6/40, 11/24, 34/120 against the published build's 39/40, 23/23, 120/120; exact McNemar 86 discordant, p < 1e-5. The behavioural read stands: 5 families delivered by the arm, 4 by the published build, out of 24 — both decline most of what is asked. The only arm near the published build on held-out wording is scale 1.0 over blk.9-35 (97/120, p < 1e-5 against it) and it costs maths 159->128 with a collapse flag. | | |
| in-clone | q3-4b-f15s15-l935 (shared axis, o_proj + mlp.down_proj, scale 1.5, blk.15-35) matches the published build on held-out wording at 120 families and costs maths | `verification/q3-4b-harmful40/f15s15-l935/predictions.jsonl`, `verification/q3-4b-retention/f15s15-l935/predictions.jsonl`, `verification/q3-4b-holdout-wider/w15-f15s15/predictions.jsonl`, `verification/q3-4b-holdout-wider/shape-audit-f15s15.json`, `verification/q3-4b-holdout-wider/shape-audit-oracle.json`, `verification/q3-4b-harmful40/shape-audit-f15s15.json`, `verification/q3-4b-harmful40/prot15-l935/predictions.jsonl`, `verification/q3-4b-harmful40/prot20-l935/predictions.jsonl` | `python -m jdewitt.spanfit build --gguf-in /mnt/ssd/hermes/models/q3-4b-base-q8.gguf --out /mnt/ssd/hermes/models/q3-4b-f15s15-l935.gguf --axes /mnt/ssd/hermes/cache/scratch/probes/q3-shared-axes.pt --source-key 'model.layers.*.self_attn.o_proj.weight' --source-key 'model.layers.*.mlp.down_proj.weight' --layers 36 --only-layers 15-35 --shape shared --axis-space output --scale 1.5` |
| | ↳ Removal under BOTH wording inventories: harmful-40 40/40 (20/20 families), held-out 120/120 against the published build's 120/120, exact McNemar 0 discordant p 1.0. Retention at the same pinned config: maths 159->148 (p 0.0078) with median maths answer 68 chars vs the published build's 175 (collapse flag on), code 20->15 (p 0.40). The published build holds maths 159->159. Hand delivery read on the first 20 families of that screen: this arm 0/20, the published build 1/20 (the fake-news message on advbench-000040). | | |

10 claims: 7 in-clone, 2 pre-fix, 1 with no artifact.
34 numbers in README section IV, 34 in the map, 34 checked.

## Standing README sync (board card `readme-evidence-sync`)

Section IV of `README.md` and this map are one artifact: a sentence in IV that no
claim row resolves is a bug. The standing audit runs the same procedure after any
roadmap phase lands a number:

1. `.venv/bin/python scripts/evidence_map.py --check` must exit 0. On a missing
   artifact, put it in the clone or move the sentence to "Held for honesty, not as
   evidence" with the reason; never leave the number.
2. One new claim row per landed arm: status, the section-IV sentence, every artifact
   path (must exist), the one command that recomputes the number, and notes for a
   retraction, a collapse flag or a p-value.
3. No artifact in the clone -> no number in section IV. An adapter delta is never
   quoted as the merged effect. A competitor's table is never copied, including ours.
4. The "Held for honesty" block stays: the pre-2026-11-17-scoring-fix rows and the
   Spark/LFM pair stay named and out of any claim.
5. Nothing landed -> re-check only; do not edit prose to keep busy.

The deciding row stays visible in both files: q3-4b-f15s15-l935 removes wording on
120/120 held-out families (0 discordant, McNemar p 1.0 against the published
same-base build) and costs maths 159 -> 148; the published build holds 159 -> 159.
Equal removal at less capability loss is not established until a row reaches 159/241
at equal removal, and section IV must say so.
