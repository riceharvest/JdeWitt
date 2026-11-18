# Instrument gates, proven by running them

Both gates were exercised against a live server on 127.0.0.1:8134 serving
`q3-4b-base-q8.gguf` (Q8_0). Commands are exact; nothing here is a claim about
model behaviour.

**A wave must serve the artifact its serve-config names.** `scripts/capability_wave.py`
now reads `/props` before generating a single row and compares `model_path` to the `-m`
path in the declared config.

```
$ python scripts/capability_wave.py verification/capability/capability-run-en.jsonl ... \
    --serve-config "-m /mnt/ssd/hermes/models/q3-4b-attn-l935.gguf ..."
port 8134 is serving /mnt/ssd/hermes/models/q3-4b-base-q8.gguf, not
/mnt/ssd/hermes/models/q3-4b-attn-l935.gguf. Refusing to record a wave against an
artifact the serve config does not name.     (exit 1)
```

This is the failure that produced a fake `40/40 removal` earlier tonight: the lane
built nothing because its axes file was missing, served a nonexistent GGUF, got empty
answers on all 40 rows, and an empty answer trips no refusal wording.

**All-invalid rows are unmeasured, never a score.** `wave.json` now carries
`"measured"`, and a wave where every row is invalid exits 1:

```
UNMEASURED: all 61 rows are 61 invalid rows. This is not a result — read it as the
screen never having happened.
```

That run is `unmeasured-because-cap64.wave.json`: `max_tokens=64` is too small for the
code rows, so all 61 came back invalid. Under the old behaviour that file would have
been summarized as `capability_passed: 0` of 61, which reads as a capability collapse
on the base model. It was a screen-capacity mistake.

`scripts/frontier_report.py` now prints `UNMEASURED (40/40 invalid)` instead of a
removal rate and a Wilson bound for a screen like the one above.
