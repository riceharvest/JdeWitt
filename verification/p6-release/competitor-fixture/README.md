# Competitor-shaped release fixture (16-bit, no adapter, no plan card)

This directory is a SHAPE FIXTURE for `jdewitt release-diff`, not a release and
not a competitor's numbers. It models the foreign layout Phase 6 is compared
against: one 16-bit GGUF, a README, an evaluation record and a files.json
index, with **no adapter, no twin record, no plan card and no signature**.

The model entry below names the real 16-bit artifact on this host by its
sha256 (`Spark-X2.6-1.7B-f16.gguf`) so the mismatch is a fact about a file we
can see, but the fixture does not ship those bytes. `evaluation.json` carries
no measured values on purpose: inventing a competitor's numbers is exactly what
this fixture exists to avoid.
