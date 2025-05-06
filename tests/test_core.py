from iosacal.core import (
    calibrate,
    combine,
    simulate,
    R,
    CalibrationCurve,
    RadiocarbonDetermination,
)

import pkg_resources

import numpy as np
from scipy import stats

import pytest


class TestCalibrate:
    def test_calibrate(self):
        assert calibrate(7345, 34, 7513.0, 13.0) == 6.50743743801389e-07
        assert calibrate(7345, 34, 7301.0, 23.0) == 0.013715329452426768

    @pytest.mark.xfail
    def test_error(self):
        with pytest.raises(ValueError):
            calibrate(7345, 0, 7513.0, 10.0)


class TestQuantiles:
    def test_Quantiles(self):
        quantiles = R(3000, 540, "test1").calibrate("intcal20").quantiles()

        print(quantiles)
        assert np.array_equal(quantiles[50], 3190.0)
        assert np.array_equal(quantiles[95], [1828.0, 4572.0])
        assert np.array_equal(quantiles[68], [2469.0, 3844.0])


class TestCalibrationCurve:
    @pytest.fixture(autouse=True)
    def curve(self):
        curve_path = pkg_resources.resource_filename("iosacal", "data/intcal20.14c")
        self.curve = CalibrationCurve(curve_path)

    def test_curve_from_curve(self):
        assert np.array_equal(CalibrationCurve(self.curve), self.curve)

    def test_calibration_curve_title(self):
        assert self.curve.title == "Atmospheric data from Reimer et al (2020)"
        assert np.array_equal(self.curve[0], np.array([54999.0, 50099.05, 1023.7]))
        assert np.array_equal(self.curve[-1], np.array([0.0, 199.0, 11.0]))


class TestCalibrationCurvePath:
    def curve_path(self):
        self.curve = CalibrationCurve("intcal20")
        assert self.curve.title == "Atmospheric data from Reimer et al (2020)"

    def curve_path_error(self):
        with pytest.raises(FileNotFoundError):
            self.curve = CalibrationCurve("non_existing_curve")


class TestRadiocarbonDetermination:
    @pytest.fixture(autouse=True)
    def r(self):
        self.r = RadiocarbonDetermination(7505, 93, "P-769")

    def test_radiocarbon_determination(self):
        assert self.r.date == 7505
        assert self.r.sigma == 93
        assert self.r.id == "P-769"

    def test_radiocarbon_calibration(self):
        r_cal = self.r.calibrate("intcal13")
        assert np.array_equal(
            r_cal[0], np.array([8.96800000e03, 2.3781370616220048e-08])
        )

    def test_radiocarbon_calibration20(self):
        r_cal = self.r.calibrate("intcal20")
        assert np.allclose(r_cal[0], np.array([8.97400000e03, 3.23557224e-08]))


class TestCombine:
    @pytest.fixture(autouse=True)
    def determinations(self):
        self.determinations = (
            R(7345, 35, "Test-A"),
            R(7387, 21, "Test-B"),
            R(7329, 40, "Test-C"),
        )

    def test_combine(self):
        r_combined = combine(self.determinations)
        print(r_combined)
        r_reference = R(
            7367.98,
            16.42,
            "Combined from Test-A, Test-B, Test-C with test statistic 2.201",
        )
        assert r_combined.date == pytest.approx(r_reference.date)
        assert r_combined.sigma == pytest.approx(r_reference.sigma, rel=1e-5)

    def test_error_combine_single(self):
        with pytest.raises(TypeError):
            combine(self.determinations[0])

    def test_error_combine_wrong_datatype(self):
        with pytest.raises(AttributeError):
            combine(
                (
                    self.determinations[0],
                    (7345, 35, "This is a tuple, not a RadiocarbonDetermination"),
                )
            )


class TestSimulate:
    def test_simulate(self):
        curve = CalibrationCurve("intcal20")
        sim = simulate(41250, 23, "simulated", "intcal20")
        real = R(41250, 23, "real")
        assert type(sim) == type(real)

    @pytest.mark.skip(reason="This test does too much and can predictably fail")
    def test_sampling(self):
        curve = CalibrationCurve("intcal20")

        # Test multiple lab errors.
        stds = [0, 11, 21]

        # Index corresponds to 700 CE in intcal20.
        idx = 53749
        assert curve[idx, 0] == 1950 - 700

        curve_mean, curve_std = curve[idx, 1:]

        for std in stds:
            samples = [
                simulate(700, std, "simulated", "intcal20", BP="ce").date
                for _ in range(100)
            ]

            expected_mean = curve_mean
            expected_std = np.sqrt(curve_std**2 + std**2).item()
            test = stats.kstest(
                samples,
                lambda x: stats.norm.cdf(x, loc=expected_mean, scale=expected_std),
            )

            assert test.pvalue > 0.01
