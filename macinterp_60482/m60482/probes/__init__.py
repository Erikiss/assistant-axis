"""Probe modules.

Importing this package registers every probe.  Each module is independent: it reads a
:class:`~m60482.measure.Bundle` and returns a
:class:`~m60482.registry.ProbeResult`.  None of them loads a model.

Run order (``order=``) is deliberate.  The premise checks run first, because if the
anchor does not meet the entry criterion for memorization at all, the expensive
mechanistic probes are answering a question nobody should be asking.
"""

from . import (  # noqa: F401
    attention_sink,
    context_dependence,
    ctx_nll_structure,
    focus_directions,
    memorization_entry,
    name_variants,
    noise_floor,
    norm_attribution,
    position_bias,
    rank_attribution,
    scope_checks,
    sink_stability,
)
