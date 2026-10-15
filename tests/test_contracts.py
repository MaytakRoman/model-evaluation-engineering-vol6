import unittest
from programs.acceptance_eval import exact_bounds, evaluate, sample_records, sample_config
from programs.agent_eval import demo as agent_demo
from programs.statistics_eval import holm_adjust, paired_cluster_bootstrap

class Contracts(unittest.TestCase):
    def test_zero_exact(self):
        lower, upper = exact_bounds(0, 300, .05)
        self.assertEqual(lower, 0)
        self.assertAlmostEqual(upper, 1 - .05 ** (1 / 300), places=11)

    def test_denominator(self):
        with self.assertRaises(ValueError):
            evaluate(sample_records()[:-1], sample_config())

    def test_trace_failure(self):
        rows = sample_records()
        rows[10]["trace_complete"] = False
        result = evaluate(rows, sample_config())
        self.assertFalse(result["checks"]["R1"])
        self.assertFalse(result["statistical_and_technical_pass"])

    def test_agent_state(self):
        result = agent_demo()
        self.assertTrue(result["good"]["success"])
        self.assertEqual(result["retry"]["tickets"], 2)
        self.assertFalse(result["early"]["success"])

    def test_holm(self):
        result = holm_adjust([.004, .018, .041, .20])
        for a, b in zip(result, [.016, .054, .082, .20]):
            self.assertAlmostEqual(a, b)

    def test_pair_reversal(self):
        rows = [dict(task_id=str(i), group_id=str(i), score_a=a, score_b=b)
                for i, (a, b) in enumerate([(0, 1), (1, 1), (1, 0), (0, 1)])]
        a = paired_cluster_bootstrap(rows, repeats=1000)
        reverse = [dict(r, score_a=r["score_b"], score_b=r["score_a"]) for r in rows]
        b = paired_cluster_bootstrap(reverse, repeats=1000)
        self.assertAlmostEqual(a["effect"], -b["effect"])
        self.assertAlmostEqual(a["lower"], -b["upper"])

if __name__ == "__main__":
    unittest.main()
