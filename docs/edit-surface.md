# Edit surface, layer profile, and strength

Three knobs decide *what* gets edited and *how hard*. They are independent, and
none of them is verified to remove refusals on a real checkpoint — the tests
below prove the operators and the validation, not refusal quality.

## `edit_targets`

A residual direction can be removed on either side of a projection.

`writeback` (default) edits `o_proj`/`out_proj` and `down_proj`. The module still
reads whatever it read, but no longer writes the direction back into the residual
stream:

```
dW = -s * d.T @ (d @ W)          needs out_features == hidden
```

`full` additionally edits `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj`.
These *read* the residual stream, so the direction is removed from their input:

```
dW = -s * (W @ d.T) @ d          needs in_features == hidden
```

Both forms remove the same component of the same vector; they differ in where.
At `s = 1` the read-side edit makes an input equal to `d` behave exactly like a
zero input, and `s = 0.5` scales that component by a half.

The read side is not cosmetic: under grouped-query attention `k_proj` and `v_proj`
write `num_key_value_heads * head_dim` values, which is smaller than the residual
width, so the write-back operator cannot even be expressed on them. With
`edit_targets='full'` they become editable, and each side is searched by its own
strength (`attn`/`mlp` for write-back, `attn_in`/`mlp_in` for read), so read and
write pressure are tuned independently rather than sharing one number.

Row normalization is explicit on both sides. Read-side `pre` is algebraically
the unnormalized read projection: normalizing and restoring a row cancels.
Read-side `full` rescales each rounded edited row to its original norm; it retains
the projected nullspace at full removal strength. Collapsed nonzero rows reject
the edit rather than creating invalid values. Full-surface LEACE uses its oblique
left/right factors on the correct side of each weight. See
[budgeted generation and operator fixes](budgeted-generation.md).

A fused expert tensor has no separable read-side operator here. Patterns that
resolve to one raise rather than being silently skipped.

## `method_options.layer_profile`

By default Optuna searches a layer-relative `center`/`width`/`floor` shape per
edit group.

`"layer_profile": "discriminative"` replaces that shape with a measurement. For
each layer it computes the good/bad separation of the captured residuals divided
by the pooled spread of those same class means:

```
score[l] = mean_family(||bad_family[l] - good_family[l]||) / pooled_spread[l]
profile  = score / max(score)        in [0, 1]
```

A layer with smaller matched-family separation relative to its within-class
spread receives less weight. This is a custom heuristic, not Abliterix's
opposite-sign selective-layer rule or a reproduced quality gain. Applied strength becomes
`strength * profile[layer]`.

Two properties matter for honesty. It *replaces* the searched shape rather than
multiplying with it, so the two are never double-counted. And it is only offered
to methods whose operator is a plain per-layer projection — `leace` and `ara`
carry their own factors, and a layer profile there would be accepted and then
ignored, so it is rejected at validation time instead.

Only one `strength` per group is searched in this mode; searching `center`,
`width` and `floor` would spend budget on values the operator does not read.

## `strength_range`

The hard bound is `MAX_STRENGTH = 6.0`, up from the previous `2.0`, because
published near-zero-refusal releases sit above the old ceiling. The bound is not
the search range: the default searched range is still `[0, 1.5]`, and a run must
ask for a wider one explicitly with `"strength_range": [lo, hi]`.

Widening the range widens the search space, so it is a deliberate trade rather
than a free improvement.

## What is not verified

- that any of this removes refusals on a real checkpoint;
- that the read side is more effective than the write-back side, or by how much;
- that the discriminative profile beats a searched shape on a real model;
- router profiling/suppression and independently fitted per-expert directions.
  The existing fused-expert adapter projects a shared layer direction across
  every expert, which already overlaps Abliterix's basic EGA operator. EGA does
  not itself require independently learned expert directions.

`b70-ablator methods` reports the same list under `not_verified`, and every
catalogued method still declares `full_model_b70_verified: false`.

## Tests

`tests/test_edit_surface.py` proves the read-side operator against its closed
form, the grouped-query reach, independent per-group strengths, the strength
bound, and (on a B70, marked `xpu`) that a 14-tensor read+write edit survives a
streamed export and reload with bit-identical logits.

`tests/test_layer_profile.py` proves the profile follows measured separation,
that a non-separating layer scores near zero, that a degenerate measurement is
rejected, that a profiled plan refuses searched-shape parameters, and the
`strength_range` and `edit_targets` validation.
