"""Check joint outcomes, conditional denominators, order pairing and resampling."""
from collections import Counter
import copy
import random
import unittest

import numpy as np

from experiments.analysis.metrics import item_results
from experiments.analysis.paired_configurations import (
    cluster_analysis, contribution, metric_counts, rate,
)
from tests.test_metrics import responses


def fixture():
    overrides = [
        {"f_then_g": "B"},
        {"g_then_f": "B", "s_then_g": "E"},
        {"g_then_s": "B"},
        {"g_then_f": "unmapped", "f_then_g": "E"},
        {"g_then_s": "E", "g_then_f": "B"},
        {"closed_book": "B"},
    ]
    items = item_results([row for i, o in enumerate(overrides) for row in responses(str(i), o)])
    for item in items:
        item["cluster_id"] = f"cluster:{item['item_id']}"
    return items


def counts(items):
    return metric_counts(np.stack([contribution(r) for r in items]).sum(axis=0))


class PairedConfigurationTests(unittest.TestCase):
    def test_fixed_cohort_all_cells_invalid_and_order_pairing(self):
        result = counts(fixture())
        expected = {
            "G_first.pair.both_gold": (1, 5),
            "G_first.pair.S_only_gold": (2, 5),
            "G_first.pair.F_only_gold": (1, 5),
            "G_first.pair.neither_gold": (1, 5),
            "G_first.F_given_S_gold.gold": (1, 3),
            "G_first.F_given_S_gold.lure": (1, 3),
            "G_first.F_given_S_gold.invalid": (1, 3),
            "misleading_first.pair.both_gold": (2, 5),
            "misleading_first.pair.S_only_gold": (2, 5),
            "misleading_first.pair.F_only_gold": (1, 5),
            "misleading_first.pair.neither_gold": (0, 5),
            "misleading_first.F_given_S_gold.gold": (2, 4),
            "misleading_first.F_given_S_gold.lure": (1, 4),
            "misleading_first.F_given_S_gold.abstain": (1, 4),
            "order_mean.S.gold": (7, 10),
            "order_mean.F.gold": (5, 10),
        }
        for key, pair in expected.items():
            self.assertEqual(tuple(int(v) for v in result[key][:2]), pair, key)

    def test_empty_conditional_denominator_is_not_zero(self):
        items = item_results(responses("0", {"g_then_s": "E", "s_then_g": "B"}))
        result = counts(items)
        self.assertTrue(np.isnan(rate(*result["G_first.F_given_S_gold.gold"][:2])))
        self.assertEqual(result["G_first.F.gold"][:2], (1, 1))

    def test_model_equal_average_is_not_pooled_conditional_rate(self):
        a = fixture()
        b = copy.deepcopy(a)
        # Retain only item 0 in model b's original baseline-correct subset.
        for r in b[1:]:
            r["A_m"] = False
        result, _, _ = cluster_analysis({"a": a, "b": b}, ["a", "b"], "cluster_id", 100, 9)
        metric = result["panel"]["G_first.F_given_S_gold.gold"]
        self.assertAlmostEqual(metric["estimate"], 100 * (1 / 3 + 1) / 2)
        self.assertNotAlmostEqual(metric["estimate"], 100 * (1 + 1) / (3 + 1))
        self.assertEqual(metric["model_denominators"], {"a": 3, "b": 1})
        self.assertGreater(metric["invalid_replicates"], 0)
        self.assertEqual(metric["valid_replicates"] + metric["invalid_replicates"], 100)

    def test_draws_keep_conditions_models_and_cluster_item_weights_paired(self):
        a = fixture()
        extra = copy.deepcopy(a[0])
        extra["item_id"] = "extra"
        a.append(extra)  # same cluster, two paired items
        result, _, draws = cluster_analysis({"a": a, "b": copy.deepcopy(a)}, ["a", "b"],
                                            "cluster_id", 20, 13)
        weights = Counter(random.Random(13).choices(range(6), k=6))
        expected_n = 2 * weights[0] + sum(weights[i] for i in range(1, 5))
        expected_s_only = weights[1] + weights[3]
        num, den, _ = draws["G_first.pair.S_only_gold"]
        self.assertEqual(int(num[0, 0]), expected_s_only)
        self.assertEqual(int(den[0, 0]), expected_n)
        np.testing.assert_array_equal(num[:, 0], num[:, 1])
        np.testing.assert_array_equal(den[:, 0], den[:, 1])
        self.assertEqual(result["models"]["a"], result["models"]["b"])

    def test_single_and_mixed_lure_reduction_share_A(self):
        items = fixture()
        # Single lure selection on the excluded sixth item must not enter.
        items[5]["outcomes"]["s_only"] = {"answer": "B", "gold": 0, "lure": 1,
                                            "abstain": 0, "other": 0, "invalid": 0}
        result = counts(items)
        self.assertEqual(result["single.S.lure"][:2], (0, 5))
        self.assertEqual(result["G_first.lure_reduction.S"][:2], (-1, 5))


if __name__ == "__main__":
    unittest.main()
