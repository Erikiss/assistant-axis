"""Does any published mechanism actually explain the state-60482 effect?

A single Pile passage entered Pythia-1.4B's training at step 131278.  Across the
checkpoints that straddle it, the probability of one held-out token (" per", in
"74 km per hour") moved by +14% relative -- and that movement was read as memorization.

This package tests that reading against the published explanations, on a Colab A100.
Each candidate mechanism becomes a probe with a decision rule written before the data
exist, referenced against 40 control passages and two placebo checkpoint boundaries.

    from m60482 import passages, measure, registry, config as C

    ps = passages.resolve(roots=["/content/drive/MyDrive/Colab_Pythia_Results"])
    bundle = measure.measure(ps, C.RunConfig(run_id="demo"))
    results = registry.run(bundle)
"""

from . import config, measure, model, passages, registry, stats  # noqa: F401

__all__ = ["config", "measure", "model", "passages", "registry", "stats"]
__version__ = "0.1.0"
