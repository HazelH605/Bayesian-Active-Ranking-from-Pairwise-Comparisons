"""Check policy state, paired streams, complete batches, and exact resumption."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from anytime_ranking.config import Config, ExperimentConfig, InferenceConfig, POLICY_SEEDS, ScenarioConfig, _validate
from anytime_ranking import experiment
from anytime_ranking.experiment import run_episode
from anytime_ranking.policies import softmax_pair_risk_probabilities
import numpy as np


def scientific(result):
    """Exclude wall-clock timing from deterministic comparisons."""
    def clean(row):
        return {k: v for k, v in row.items() if not k.endswith("_seconds") and not (
            isinstance(v, float) and np.isnan(v))}
    return clean(result.run), [clean(row) for row in result.trajectory]


class SimulationTests(unittest.TestCase):
    def test_softmax(self):
        risks = np.array([.05, .2, .4])
        np.testing.assert_array_equal(softmax_pair_risk_probabilities(risks, 0), np.ones(3)/3)
        probabilities = softmax_pair_risk_probabilities(risks, 30)
        self.assertTrue(np.all(np.diff(probabilities) > 0))
        self.assertAlmostEqual(probabilities.sum(), 1.)
        self.assertTrue(np.all(softmax_pair_risk_probabilities(risks, 1e300) > 0))

    def test_all_policies_resume(self):
        class Interrupted(Exception):
            pass

        for initial in (0, 1):
            for batch in (1, 2):
                for cuts in ((2, 4), (1, 2, 4), (1, 2, 3, 4)):
                    policies = tuple(p for p in POLICY_SEEDS if len(cuts) == 2 or p not in {"hamming_lucb", "topk_aware"})
                    scenario = ScenarioConfig("small", 4, cuts, truth_mode="prior_predictive", pairing_id=602)
                    config = Config(InferenceConfig(posterior_samples=64, acquisition_samples=32),
                        ExperimentConfig(policies=policies, replicates=1, max_comparisons=6*batch,
                            evaluation_budgets=(6*batch,), initial_comparisons_per_pair=initial,
                            batch_size=batch, jobs=1, checkpoint_interval=2), (scenario,))
                    _validate(config)
                    for index, policy in enumerate(policies):
                        with self.subTest(initial=initial, batch=batch, cuts=cuts, policy=policy):
                            complete = run_episode(config, scenario, policy, 0, index, 0)
                            with TemporaryDirectory() as folder:
                                path = Path(folder)/"query.json"
                                save = experiment.atomic_json

                                def interrupt_after_save(file, value):
                                    save(file, value)
                                    if value["comparisons"] == 2*batch:
                                        raise Interrupted()

                                with patch.object(experiment, "atomic_json", interrupt_after_save):
                                    with self.assertRaises(Interrupted):
                                        run_episode(config, scenario, policy, 0, index, 0, query_checkpoint_path=path)
                                resumed = run_episode(config, scenario, policy, 0, index, 0, query_checkpoint_path=path)
                            self.assertEqual(scientific(complete), scientific(resumed))

    def test_policy_order_keeps_random_stream(self):
        scenario = ScenarioConfig("small", 4, (2, 4), pairing_id=601)
        config = Config(InferenceConfig(posterior_samples=64, acquisition_samples=32),
                        ExperimentConfig(replicates=1, max_comparisons=6, evaluation_budgets=(6,), jobs=1), (scenario,))
        original = run_episode(config, scenario, "softmax_pair_risk", 0, 0, 0)
        reordered = replace(config, experiment=replace(config.experiment, policies=("uniform", "softmax_pair_risk")))
        self.assertEqual(scientific(original), scientific(run_episode(reordered, scenario, "softmax_pair_risk", 0, 1, 0)))


if __name__ == "__main__":
    unittest.main()
