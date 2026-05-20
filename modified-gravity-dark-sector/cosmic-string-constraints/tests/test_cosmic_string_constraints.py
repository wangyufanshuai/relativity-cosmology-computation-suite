from cosmic_string_constraints import StringNetwork, amplitude_proxy, excluded_by_pta_limit


def test_amplitude_scales_with_tension():
    low = amplitude_proxy(StringNetwork(1e-12))
    high = amplitude_proxy(StringNetwork(1e-10))
    assert high > low


def test_exclusion_threshold():
    assert excluded_by_pta_limit(StringNetwork(1e-10), 1e-16)
    assert not excluded_by_pta_limit(StringNetwork(1e-14), 1e-13)
