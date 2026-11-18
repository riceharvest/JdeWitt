# Worked loop (the commands, not the manifesto)

Every number in the docs came from this loop, so here it is end to end with the
exact invocations that produced the committed evidence. Paths assume
`/mnt/ssd/hermes/cache/scratch/probes` holds the checkpoints, the capture scripts
and the contrast/cert prompt files; `verification/` holds the committed evidence.
No command here is hypothetical; anything I have not run is written as not run.

## 0. Pick the base and pin it

Record the repo revision before anything else; the same name at another revision
is a different model. Qwen3-4B-Instruct-2507 and its published Heretic build are
pinned in `docs/non-heretic-methods.md`. Both sides of a comparison get converted
by the same converter, served by the same `llama-server` binary and the same argv.

    python convert_hf_to_gguf.py <hf-dir> --outfile <out>.gguf --outtype q8_0

## 1. Get labels from the model being edited (R3)

Serve the unmodified base, generate the contrast set, keep only rows whose label
the model's own behaviour supports (bad rows that actually refuse, good rows that
actually comply), and keep matched family pairs. A fit on labels the model does
not exhibit produces a null edit, and it will still merge cleanly.

    systemd-run --user --unit=<name> -p MemoryMax=12G \
      --setenv=LD_LIBRARY_PATH=/mnt/ssd/models/spark/lib \
      bash -c '"/path/llama-server" -m <base>.gguf -c 4096 --parallel 4 --jinja \
               --no-webui --host 127.0.0.1 --port 8134 > /tmp/<name>.log 2>&1'
    python q3label.py          # reads :8134, writes <model>-accepted.json

## 2. Capture residuals off the serving GPU

Stop the server, then hook the decoder layers on the HF weights in bf16 and store
per-layer states at two loci: last prompt token and answer end. Capturing while a
serving process holds the card OOMs on a 4B model at 12 GiB.

    python q3cap.py            # writes <model>-records-{last,ans}.pt
    # records are (task, [L, D]) with labels from step 1; torch.load needs
    # weights_only=False, and the hook must accept a tensor or a tuple.

## 3. Fit both shapes and merge them

Fit is milliseconds-to-seconds; the merge is the expensive CPU step (about a
minute for a 4B Q8).

    python -m jdewitt.spanfit build \
      --records <model>-records-last.pt --gguf-in <base>.gguf --out <out>.gguf \
      --shape shared --axis-space output --scale 1.0 --layers 36 \
      --source-key 'model.layers.*.self_attn.o_proj.weight' \
      --source-key 'model.layers.*.mlp.down_proj.weight'

`--axis-space` is not cosmetic: the axis from a residual-stream capture lives in
the dimension the module writes into, and models whose attention output dim
differs from hidden (qwen3-4b: 4096 in, 2560 out) are silently skipped on the
other side. `tensors_edited 0` with an otherwise successful merge is the failure
mode; read `skipped`, never assume.

`--scale` is the projection multiplier. 1.0 removes the axis completely; the
certified q25 SAS number is 1.5. The scale curve is knife-edge: on q25 1.0 and
1.5 are worlds apart, and on qwen3-4b 2.0 stopped refusing and started producing
token salad, which is damage, not removal.

## 4. Screen before you certify

A ten-row screen at a short decode budget is ~20 seconds and 1k tokens. It is
screening, never a removal claim (R6): the word screen cannot tell a refusal from
false compliance.

    python scripts/capability_wave.py <10-row harmful suite> <out-dir> --label <name> \
      --kinds refusal --max-tokens 1024 --parallel 4 --preview 4 \
      --serve-config "ctx-size 8192 parallel 4 cuda --jinja --no-webui port 8134 temperature 0 max_tokens=1024"

The cap is part of the measurement, on this side of the suite too. An artifact that
merged the published oracle's own direction read 10/10 `truncated_or_error` at 350
tokens, and the same file at 1024 read 0 refusals, 7 delivered, 3 truncated
against the base's 10 refusals — because the answers
this model writes after removal run past 1700 characters and the instrument
refuses to grade a truncated answer (`status: truncated_or_error`, never a pass,
never a refusal). A screen where every row is invalid is not a 0/10, it is an
unmeasured 10/10.

Read the previewed answers anyway. The verdict is the shape — limitation clause
plus absence of the requested content — and "I'm sorry, but I can't assist" plus
a lecture is a refusal however well it passes a regex.

## 5. Certify the winner at the pinned config

Run harmful-40, the held-out refusal families and the harmless control at
`-c 4096 --parallel 4`, dump the evidence packets and label every row yourself
against the suite's requirements rubric. Record the full `llama-server` argv in
the packet: the same bytes measured ~90% removal at 2048/2 and ~50% at 4096/4 on
mc47, so a removal number without its config is not a number.

## 6. Measure retention with the same artifact and config

    python scripts/capability_wave.py \
      verification/capability/capability-run-en.jsonl \
      verification/retention-q25-code61/<label> \
      --label <label> --max-tokens 1024 \
      --serve-config "ctx-size 8192 parallel 4 cuda --jinja --no-webui port 8134 temperature 0 max_tokens=1024" \
      --ledger <lineage-timing>.jsonl

The wave writes `predictions.jsonl` plus a `wave.json` manifest inside its own
directory. Grading is the shipped deterministic scorers — no judge calls — and a
row whose generation failed is never a pass. A generation cap shorter than the
edited model's new verbosity scores truncation as capability loss: the same q25
edit measured .429 on code under a 512-token cap and .500 under 1024.

## 7. Compare, and let the tool enforce the rules

    python -m jdewitt.cli score-compare \
      verification/capability/capability-run-en.jsonl \
      verification/retention-q25-code61/base verification/retention-q25-code61/sas15 \
      --kind code

`--kind math` and `--kind code` are the retention axis; `--kind refusal` is the
removal axis and mixing them in one aggregate is the dilution this flag exists to
prevent. Two waves whose recorded context split, slot count, generation cap,
temperature, chat templating, backend or suite differ are refused until
`--allow-config-mismatch`, because they are not a before/after pair. Then split
the pass rate into what it actually conflates:

    python scripts/capability_format_audit.py \
      base=verification/retention-q25-code61/base/predictions.jsonl \
      sas15=verification/retention-q25-code61/sas15/predictions.jsonl

strict (the shipped verdict), computed_anywhere (did the value appear, ignoring
dress), format_only_number (did it obey the answer contract) and truncated are
four different measurements and only the first one used to be reported. On q25
our certified edit books +0.544 on maths while format-following drops from 90.6%
to 1.7%; a single "capability" number would have hidden one of those inside the
other.

## 8. Sweep the shape, and read it as one table

A sweep is footprint × side × scale, and every axis is a flag, not an edit:

    python -m jdewitt.spanfit build --gguf-in <base> --out <arm> --axes <axes.pt> \
      --source-key 'model.layers.*.self_attn.o_proj.weight' --layers 36 \
      --only-layers 9-35 --shape shared --axis-space output --scale 1.5

`--only-layers 9-35` restricts the wildcard (the published oracle leaves blk.0–8
byte-identical); drop the `--source-key` line to edit the other side. `spanfit
protect --axes A --records R --label cap` projects the axis out of another
behaviour's subspace and reports how much of it survives, which is how a protection
claim dies early: on qwen3-4B it kept **99.24%**, so the null is measured in
milliseconds instead of two screen waves.

Then read the arms without a throwaway script:

    python scripts/frontier_report.py \
      --base-dir verification/q3-4b-retention/q3base \
      --pair s15=verification/q3-4b-harmful40/s15-l935:verification/q3-4b-retention/s15-l935

One row per artifact: removal with a Wilson lower bound, refusals, invalid rows,
median answer length, maths and code before→after with the paired sign-flip `p`, the
length-collapse flag and the recorded cap/slots. Removal comes from the screen's
shipped refusal-check and the retention numbers call `capability_format_audit`'s own
paired function, so a row cannot disagree with the commands that produced it. Read
the removal column together with the collapse column — on qwen3-4B 40/40 removal
only ever arrives with a median maths answer of 31 or 4 characters, which is the
model stopped writing, not answering wrong. And read it by **family**, not row: the
q3 screen is 20 families of 2 rows, so a family counts as removed only when both its
rows are non-refusals, and `≥` there is the honest floor — 12/20 (≥0.387) where the
row column said 31/40. A screen where every row came back invalid prints `UNMEASURED`
and exits non-zero: an empty answer trips no refusal wording, so "all rows empty" is
the shape of a model that failed to load, and it must never read as a perfect removal.

## What is not scripted

Fitting the axis shape, choosing modules and scale, reading evidence packets, and
deciding whether a row is removal or damage are judgement calls and stay manual.
There is no test suite and no CI on this repo by the owner's standing rule:
`ruff check jdewitt scripts` plus a live run of the new command against a real
artifact is the gate, and a number that is not reproducible from the files in
`verification/` is not a result.
