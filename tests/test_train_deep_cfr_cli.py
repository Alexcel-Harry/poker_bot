import tempfile
import unittest
from pathlib import Path

from examples.train_deep_cfr import build_parser, parse_hidden, run_from_args


class TrainDeepCFRCliTests(unittest.TestCase):
    def test_parser_and_hidden_validation(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.game, "kuhn")
        self.assertEqual(args.iterations, 100)
        self.assertEqual(args.seats, 3)
        self.assertEqual(parse_hidden("64, 32"), (64, 32))
        with self.assertRaises(ValueError):
            parse_hidden("64,0")

    def test_small_kuhn_run_writes_resumable_and_inference_artifacts(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser.parse_args(
                [
                    "--iterations", "1",
                    "--traversals-per-player", "2",
                    "--hidden", "8",
                    "--advantage-train-steps", "1",
                    "--strategy-train-steps", "1",
                    "--batch-size", "4",
                    "--device", "cpu",
                    "--snapshot-out", str(root / "snapshot.pt"),
                    "--policy-out", str(root / "policy.pt"),
                    "--summary-out", str(root / "summary.json"),
                    "--log-every", "0",
                ]
            )

            result = run_from_args(args)

            self.assertEqual(result["algorithm"], "external_sampling_deep_cfr")
            self.assertTrue((root / "snapshot.pt").exists())
            self.assertTrue((root / "policy.pt").exists())
            self.assertTrue((root / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
