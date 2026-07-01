import unittest

from examples.headless_demo import run_demo


class DemoTests(unittest.TestCase):
    def test_headless_demo_runs_seeded_three_player_session(self):
        histories = run_demo(hands=2, seed=11)

        self.assertEqual(len(histories), 2)
        self.assertTrue(all(history["events"] for history in histories))


if __name__ == "__main__":
    unittest.main()
