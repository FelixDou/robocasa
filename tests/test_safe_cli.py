import contextlib
import io
import unittest

from tests.safe_import_helper import install_lightweight_robocasa_packages

install_lightweight_robocasa_packages()

from robocasa.recovery.safe import (
    collect_atomic_rollouts,
    collect_rollouts,
    conformal,
    dataset,
    evaluate,
    export_to_official_safe,
    plots,
    train,
    validate_atomic_dataset,
)


class TestSafeCLIHelp(unittest.TestCase):
    def test_all_cli_help_exits_cleanly(self):
        for module in (
            collect_atomic_rollouts,
            validate_atomic_dataset,
            export_to_official_safe,
            collect_rollouts,
            dataset,
            train,
            conformal,
            evaluate,
            plots,
        ):
            with self.subTest(module=module.__name__):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    with self.assertRaises(SystemExit) as exit_context:
                        module.main(["--help"])
                self.assertEqual(exit_context.exception.code, 0)
                self.assertIn("usage:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
