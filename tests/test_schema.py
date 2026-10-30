from focuswatch_dataset import schema as S


def test_every_quantity_has_columns_and_a_unit():
    for q in S.Quantity:
        assert S.COLUMNS[q], q
        assert S.UNITS[q], q


def test_accel_bands_do_not_overlap_and_leave_a_forbidden_gap():
    lo_hi_user, lo_hi_total = S.ACCEL_USER_BAND, S.ACCEL_TOTAL_BAND
    assert lo_hi_user[1] < lo_hi_total[0]
    # Why: the gap is the detector. A user/total mix-up averages into it.
    assert (lo_hi_user[1], lo_hi_total[0]) in S.ACCEL_FORBIDDEN_BANDS


def test_ms2_band_is_forbidden():
    # A forgotten division by 9.80665 lands near 9.8.
    assert any(lo <= 9.80665 <= hi for lo, hi in S.ACCEL_FORBIDDEN_BANDS)


def test_quaternion_columns_are_scalar_last():
    assert S.COLUMNS[S.Quantity.QUAT] == ("quat_x", "quat_y", "quat_z", "quat_w")
