from h0_tension import Constraint, combined_constraint, tension_sigma, tension_summary


def test_tension_sigma_matches_gaussian_rule():
    a = Constraint("local", 73.0, 1.0)
    b = Constraint("early", 67.0, 1.0)
    assert round(tension_sigma(a, b), 6) == round(6.0 / 2**0.5, 6)


def test_combined_constraint_has_smaller_error():
    combo = combined_constraint("combined", [Constraint("a", 70.0, 2.0), Constraint("b", 72.0, 2.0)])
    assert combo.mean == 71.0
    assert combo.sigma < 2.0
    assert tension_summary(Constraint("l", 73.0, 1.0), Constraint("e", 67.0, 1.0))["delta_h0"] == 6.0
