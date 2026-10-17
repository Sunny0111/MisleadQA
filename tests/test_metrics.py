"""Regression tests for paper denominators, paired resampling and eligibility."""
from collections import Counter
import copy
import random
import unittest

from experiments.analysis.analyze import attach_groups
from experiments.analysis.metrics import OUTCOMES, item_results, paired_bootstrap, statistics
from experiments.main_eval.protocol import CONDITION_IDS


def responses(item_id, overrides=None):
    answers = {condition: "A" for condition in CONDITION_IDS}
    answers.update(overrides or {})
    return [{"item_id": item_id, "condition_id": condition,
             "answer_key": {"gold": "A", "lure": "B", "abstain": "E"},
             "source": "source", "dimension": "time", "analysis_group_id": f"dep:{item_id}",
             "evaluation_eligible": True, "finish_reason": "stop", "response": answers[condition]}
            for condition in CONDITION_IDS]


def twenty_clusters():
    items = item_results([row for i in range(20)
                          for row in responses(str(i), {"f_only": "B" if i == 0 else "A",
                                                        "g_then_f": "B" if i == 0 else "A"})])
    for item in items:
        item["cluster_id"] = f"c{int(item['item_id']):02d}"
    return items


class PaperMetricsTests(unittest.TestCase):
    def test_intersection_and_invalid_denominators(self):
        rows = (responses("1", {"f_only": "B", "g_then_f": "B"})
                + responses("2", {"closed_book": "C", "g_then_f": "B", "f_then_g": "B"})
                + responses("3", {"g_only": "C", "f_only": "E", "g_then_f": "B", "f_then_g": "B"})
                + responses("4", {"f_only": "unmapped", "g_then_f": "C",
                                   "g_then_s": "E", "s_then_g": "unmapped"}))
        result = statistics(item_results(rows))
        self.assertEqual((result["B_m_size"], result["G_correct_size"], result["A_m_size"]), (3, 3, 2))
        self.assertEqual(result["metrics"]["LR_F"]["denominator"], 3)
        self.assertAlmostEqual(result["metrics"]["LR_F"]["rate"], 1 / 3)
        self.assertEqual(result["metrics"]["DR_F"]["denominator"], 4)
        self.assertEqual(result["metrics"]["DR_F"]["rate"], 0.25)
        self.assertEqual(result["contrasts"]["Delta_DR"]["percentage_points"], 25)
        self.assertEqual(result["metrics"]["DR_S.invalid"]["rate"], 0.25)
        self.assertEqual(result["metrics"]["DR_S.abstain"]["rate"], 0.25)
        self.assertEqual(result["metrics"]["DR_F.gold"]["rate"], 0.5)
        for prefix in ("DR_F", "DR_S"):
            self.assertEqual(sum(result["metrics"][f"{prefix}.{outcome}"]["rate"] for outcome in OUTCOMES), 1)

    def test_empty_cohorts_are_null(self):
        result = statistics(item_results(responses("1", {"closed_book": "E", "g_only": "C"})))
        self.assertIsNone(result["metrics"]["LR_F"]["rate"])
        self.assertIsNone(result["metrics"]["DR_S"]["rate"])
        self.assertIsNone(result["contrasts"]["Delta_DR"]["percentage_points"])

    def test_resistance_uses_same_cohort_and_order_sign_is_explicit(self):
        # Item 2 is in B but not A. Its lure answer must not inflate resistance.
        rows = (responses("1", {"f_only": "B", "g_then_f": "B", "f_then_g": "A"})
                + responses("2", {"g_only": "C", "f_only": "B"})
                + responses("3"))
        result = statistics(item_results(rows))
        self.assertAlmostEqual(result["metrics"]["LR_F"]["rate"], 2 / 3)
        self.assertEqual(result["metrics"]["Single_F.on_A.lure"]["rate"], 0.5)
        self.assertEqual(result["metrics"]["DR_F"]["rate"], 0.25)
        self.assertEqual(result["contrasts"]["Resistance_F"]["percentage_points"], 25)
        self.assertEqual(result["contrasts"]["Order_F"]["percentage_points"], 50)
        self.assertEqual(result["contrasts"]["Resistance_F.g_then_f"]["percentage_points"], 0)
        self.assertEqual(result["contrasts"]["Resistance_F.f_then_g"]["percentage_points"], 50)

    def test_retests_and_incomplete_items_rejected(self):
        rows = responses("1")
        with self.assertRaisesRegex(ValueError, "first responses"):
            item_results(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            item_results(rows[:-1])

    def test_clusters_join_from_dataset_without_item_fallback(self):
        items = item_results(responses("1"))
        dataset = [{"id": "1", "source": "source", "dimension": "time",
                    "review": {"evaluation_eligible": True},
                    "grouping": {"analysis_group_id": "dep:1", "cluster_id": "source:1"}}]
        self.assertEqual(attach_groups(items, dataset)[0]["cluster_id"], "source:1")
        bad = copy.deepcopy(dataset)
        bad[0]["grouping"].pop("cluster_id")
        with self.assertRaisesRegex(ValueError, "Missing source cluster"):
            attach_groups(items, bad)

    def test_paired_draw_and_fixed_eligible_panel(self):
        a = twenty_clusters()
        b = copy.deepcopy(a)
        excluded = copy.deepcopy(a)
        excluded[-1]["A_m"] = False
        summary, draws = paired_bootstrap({"b": b, "a": a, "excluded": excluded}, replicates=50, seed=9)
        self.assertEqual(summary["panel_models"], ["a", "b"])
        self.assertEqual(summary["model_eligibility"]["excluded"]["A_m_clusters"], 19)
        for draw in draws:
            self.assertEqual(draw["models"]["a"], draw["models"]["b"])
            self.assertEqual(draw["panel"], draw["models"]["a"])
        second = paired_bootstrap({"a": a, "b": b, "excluded": excluded}, replicates=50, seed=9)
        self.assertEqual((summary, draws), second)

    def test_cluster_bootstrap_preserves_item_weighting(self):
        items = twenty_clusters()
        base = next(item for item in items if item["item_id"] == "0")
        for number in range(3):
            extra = copy.deepcopy(base)
            extra["item_id"] = f"extra{number}"
            items.append(extra)
        summary, draws = paired_bootstrap({"m": items}, replicates=5, seed=13)
        self.assertAlmostEqual(summary["panel_contrasts"]["LR"]["estimate_pp"], 100 * 4 / 23)
        weights = Counter(random.Random(13).choices(range(20), k=20))
        expected = 100 * 4 * weights[0] / (4 * weights[0] + sum(weights[i] for i in range(1, 20)))
        self.assertAlmostEqual(draws[0]["panel"]["LR"], expected)
        self.assertAlmostEqual(draws[0]["panel"]["DR"], expected / 2)

    def test_no_eligible_panel_and_inconsistent_membership(self):
        items = twenty_clusters()
        for item in items:
            item["cluster_id"] = "one_cluster"
        result, draws = paired_bootstrap({"m": items}, replicates=5)
        self.assertEqual(result["panel_models"], [])
        self.assertIsNone(result["panel_contrasts"]["LR"]["estimate_pp"])
        self.assertEqual(draws, [])
        bad = copy.deepcopy(items)
        bad[0]["cluster_id"] = "different"
        with self.assertRaisesRegex(ValueError, "memberships differ"):
            paired_bootstrap({"a": items, "b": bad}, replicates=5)


if __name__ == "__main__":
    unittest.main()
