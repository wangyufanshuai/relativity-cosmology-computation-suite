from standard_sirens import SirenEvent, estimate_h0, h0_from_event


def test_h0_estimate_for_single_low_z_event():
    event = SirenEvent("toy", redshift=0.01, luminosity_distance_mpc=42.827494, distance_sigma_mpc=4.0)
    h0, sigma = h0_from_event(event)
    assert round(h0, 1) == 70.0
    assert sigma > 0


def test_combined_estimate_between_inputs():
    events = [
        SirenEvent("a", 0.01, 42.827494, 4.0),
        SirenEvent("b", 0.02, 85.654988, 4.0),
    ]
    mean, sigma = estimate_h0(events)
    assert round(mean, 1) == 70.0
    assert sigma < h0_from_event(events[0])[1]
