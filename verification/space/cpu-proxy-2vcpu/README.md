# The 2-vCPU CPU proxy

This directory holds what `verification/space/cpu-proxy-2vcpu.json` was measured
with, so the number can be re-made on another host:

    python verification/space/cpu-proxy-2vcpu/make_report.py \
        --gguf <Qwen3-4B-Instruct-2507-Q4_K_M.gguf> \
        --bank verification/space/cpu-proxy-2vcpu/bank \
        --run run1=<time-run1.txt>,<report-run1.json> \
        --run run2=<time-run2.txt>,<report-run2.json> \
        --run run3=<time-run3.txt>,<report-run3.json> \
        --out verification/space/cpu-proxy-2vcpu.json \
        --repo unsloth/Qwen3-4B-Instruct-2507-GGUF \
        --revision a06e946bb6b655725eafa393f4a9745d460374c9 \
        --sha256 3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597

`bank/` is a **timing fixture**, not a bank entry: the shipped `spaces/bank` is
empty on purpose (§5.2 — the only measured 4B arm's axis file is not in this
clone), so this entry carries a seeded synthetic unit axis and
`measurement.state = synthetic_axis`. Its geometry, footprint
(`blk.15-35`, 42 tensors, both residual-write kinds), per-kind strength (1.5) and
source quant types are the ones the real §5.1 entry for
`qwen3-4b-instruct-2507/f15s15-l935` states, because those are what the wall
clock depends on. Its content is not an edit anyone measured:

    python verification/space/fixtures/make_real_apply_bank.py \
        --gguf <artifact.gguf> --bank verification/space/cpu-proxy-2vcpu/bank \
        --layers 15 35 --seed 1 --strength 1.5 --entry-key \
        qwen3-4b-instruct-2507/f15s15-l935-timing-fixture

The entry's `min_cos` is 0.7 rather than the shipped 0.95. A seeded random axis
produces a much smaller per-element delta than a fitted sign-aligned axis, so the
worst tensor measures cos 0.7616 at F16 — below 0.95 for a reason that is the
fixture, not the codec. The ladder still probes `Q6_K -> Q8_0 -> F16` and writes
F16, which is what the real arm's numbers imply (its Q8_0 realized cos was
0.7255). The floor changes the type that is written; it does not change how many
parameters are decoded or re-encoded at that type.

The delivered cos in the report is therefore a **codec fidelity** number on a
synthetic edit, not a removal measurement. `removal_measured` is false.
