# P7 — reproduction index on the real Spark release

`EVIDENCE.md` is the measured record; `gates/` holds the raw stdout of each
command. The release directory itself (1.1 GB of model bytes) is not in git.

Rebuild the sealed copy from the P6 release dir:

```sh
P6="/home/dario/.agentboard/worktrees/t_a44a4d22/verification/releases/spark-1.7b-q4km-p6"
DIR=/tmp/release-spark-p7
cp -a --reflink=auto "$P6" "$DIR"
uv run jdewitt release-repro "$DIR" --out "$DIR/repro.json"
uv run jdewitt release-sign "$DIR" --secret /path/to/release-signing-key.pem
uv run jdewitt release-verify-signature "$DIR" \
  --public verification/p6-release/keys/release-signing-key.pub.pem
uv run jdewitt release-repro "$DIR" --check
```

The base artifact is the user's quant, not shipped: `--check` reads it from
`--base PATH`, a `base.gguf` inside the release, or the path `evaluation.json`
declares. Without the signing secret, `--out` + `--check` still work against
`repro.json` as the unsigned seal.
