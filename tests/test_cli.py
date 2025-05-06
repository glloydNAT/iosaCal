from click.testing import CliRunner
from iosacal.cli import main


def test_iosacal_cli():
    runner = CliRunner()
    result = runner.invoke(main, ["-d", "1100", "-s", "45", "--id", "TEST"])
    assert result.exit_code == 0
    assert "iosacal" in result.output.lower()


def test_iosacal_missing_option():
    runner = CliRunner()
    result = runner.invoke(main, ["-d", "1100", "-s", "45"])
    assert result.exit_code != 0


def test_iosacal_equal_number():
    runner = CliRunner()
    result = runner.invoke(
        main, ["-d", "1100", "-s", "45", "--id", "TEST", "-d", "1080"]
    )
    assert result.exit_code != 0
    assert "equal" in result.output


def test_iosacal_plot():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["-d", "1100", "-s", "45", "--id", "TEST", "-p"])
        assert result.exit_code == 0


def test_iosacal_plot_stacked():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            [
                "-d",
                "1100",
                "-s",
                "45",
                "--id",
                "TEST1",
                "-d",
                "1080",
                "-s",
                "35",
                "--id",
                "TEST2",
                "-p",
                "-m",
            ],
        )
        assert result.exit_code == 0
