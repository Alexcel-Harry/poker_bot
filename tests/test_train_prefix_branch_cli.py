import os
import tempfile
import unittest
from pathlib import Path

from examples.train_prefix_branch import (
    build_parser,
    configure_visible_gpus,
    parse_gpu_ids,
    run_from_args,
)


class TrainPrefixBranchCliTests(unittest.TestCase):
    def test_parse_gpu_ids_accepts_single_and_multiple_ids(self):
        self.assertEqual(parse_gpu_ids("0"), (0,))
        self.assertEqual(parse_gpu_ids("0,1,3"), (0, 1, 3))
        self.assertEqual(parse_gpu_ids(" none "), ())
        self.assertEqual(parse_gpu_ids(""), ())

    def test_parse_gpu_ids_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            parse_gpu_ids("-1")
        with self.assertRaises(ValueError):
            parse_gpu_ids("0,gpu1")

    def test_configure_visible_gpus_sets_environment_mapping(self):
        env = {}

        configure_visible_gpus((0, 2), env)

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,2")

    def test_configure_visible_gpus_hides_cuda_when_none_requested(self):
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}

        configure_visible_gpus((), env)

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")

    def test_cli_runs_small_training_and_writes_summary(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "summary.json"
            args = parser.parse_args(
                [
                    "--iterations",
                    "1",
                    "--branch-width",
                    "2",
                    "--branch-depth",
                    "1",
                    "--integer-action-budget",
                    "4",
                    "--max-actions-per-episode",
                    "2",
                    "--starting-stack",
                    "200",
                    "--device",
                    "cpu",
                    "--gpus",
                    "none",
                    "--output",
                    str(output),
                ]
            )

            result = run_from_args(args, env={})

            self.assertTrue(result["prefix_branch"])
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["gpu_ids"], [])
            self.assertGreater(result["training_samples"], 0)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
