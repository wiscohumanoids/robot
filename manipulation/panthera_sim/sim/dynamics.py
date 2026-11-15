"""Names for recording and deployment physics (no simulator dependency)."""

CONTACT_DYNAMICS = "contact-v2"
LEGACY_DYNAMICS = "weld-v1"
DYNAMICS_MODES = (CONTACT_DYNAMICS, LEGACY_DYNAMICS)


def recorded_dynamics(metadata: dict) -> str:
    """Untagged recordings predate physical grasping and used the weld."""
    mode = metadata.get("simulation_dynamics", LEGACY_DYNAMICS)
    if mode not in DYNAMICS_MODES:
        raise ValueError(f"Unknown simulation dynamics: {mode!r}")
    return mode


def provenance_dynamics(provenance: dict) -> str:
    """Validate a dataset's source tags, including portable rendered provenance."""
    modes = {recorded_dynamics(source) for source in provenance.get("sources", {}).values()}
    if "simulation_dynamics" in provenance:
        modes.add(recorded_dynamics(provenance))
    if len(modes) > 1:
        raise ValueError("Dataset mixes simulation dynamics; export each version separately")
    return next(iter(modes), LEGACY_DYNAMICS)
