"""Compute Bayes reports, ranking risk, and credibility from a Gaussian posterior."""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil, log2
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
from .partitions import IntArray, validate_cut_points, report_from_scores, _sample_report_labels, consensus_score
from .report_optimizer import ReportCertificate, _sample_optimal_report


@dataclass(frozen=True)
class PosteriorSummary:
    risk: float
    coverage: float
    labels: IntArray
    report_certificate: ReportCertificate | None = None


class GaussianRankingEvaluator:
    """Gaussian weak-order and coverage integrals using fixed scrambled Sobol draws.

    For full rankings, pairwise weak-order entries are analytic Gaussian
    probabilities. Joint credible coverage still requires integration.
    """

    def __init__(self, n_models, n_samples, seed):
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        exponent = int(ceil(log2(n_samples)))
        sobol = qmc.Sobol(d=n_models, scramble=True, seed=seed)
        uniforms = sobol.random_base2(exponent)[:n_samples]
        uniforms = np.clip(uniforms, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        self.standard_normals = ndtri(uniforms)
        self.n_samples = int(n_samples)
        self.n_models = int(n_models)
        # Fixed integration identities make report certificates reusable after resuming.
        self.evaluator_id = (int(n_models), int(n_samples), int(seed))
        self.pair_i, self.pair_j = np.triu_indices(n_models, 1)

    @staticmethod
    def _factor(covariance):
        covariance = 0.5 * (covariance + covariance.T)
        try:
            return np.linalg.cholesky(
                covariance + 1e-12 * np.eye(covariance.shape[0])
            )
        except np.linalg.LinAlgError:
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values, 1e-12)
            return vectors @ np.diag(np.sqrt(values))

    def samples(self, belief):
        # Reuse standard normal integration points across posterior states and methods.
        if belief.n_models != self.n_models:
            raise ValueError("belief dimension does not match evaluator")
        factor = self._factor(belief.covariance)
        return belief.mean + self.standard_normals @ factor.T

    def _full_ranking_report(self, belief, cut_points):
        labels = report_from_scores(belief.mean, cut_points)
        return labels, self._full_ranking_risk(belief, labels)

    def _full_ranking_risk(self, belief, labels):
        i, j = self.pair_i, self.pair_j
        variance = (
            belief.covariance[i, i] + belief.covariance[j, j]
            - 2.0 * belief.covariance[i, j]
        )
        # scipy.stats.norm.cdf delegates to this same standard-normal kernel;
        # using it directly avoids repeated distribution argument processing.
        probabilities = ndtr(
            np.asarray(
                (belief.mean[i] - belief.mean[j]) / np.sqrt(np.maximum(variance, 1e-15)),
                dtype=np.float64,
            )
        )
        weak_order = np.zeros((self.n_models, self.n_models))
        weak_order[i, j] = probabilities
        weak_order[j, i] = 1.0 - probabilities
        q = self.n_models * (self.n_models - 1) // 2
        return float((q - consensus_score(labels, weak_order)) / q)

    def pairwise_risks(self, belief, cut_points, report):
        """Expected pair losses in upper-triangle order; their mean is risk.

        Reuse the current report's exact sample sufficient statistics when
        available. This avoids drawing/ranking samples or optimizing the report
        a second time. Grouped fallback uses this evaluator's same Sobol design.
        """
        # Reuse the current Bayes report when computing SPR scores.
        validate_cut_points(cut_points, self.n_models)
        if belief.n_models != self.n_models or report.labels.shape != (self.n_models,):
            raise ValueError("belief/report dimension does not match evaluator")
        i, j = self.pair_i, self.pair_j
        relation = np.sign(report.labels[j] - report.labels[i])
        if len(cut_points) == self.n_models:
            variance = (belief.covariance[i, i] + belief.covariance[j, j]
                        - 2.0 * belief.covariance[i, j])
            return ndtr(-relation * (belief.mean[i] - belief.mean[j])
                        / np.sqrt(np.maximum(variance, 1e-15)))

        certificate = report.report_certificate
        reusable = (
            certificate is not None
            and certificate.evaluator_id == self.evaluator_id
            and certificate.n_samples == self.n_samples
            and certificate.cut_points == cut_points
            and np.array_equal(certificate.labels, report.labels)
        )
        if reusable and certificate.feature_kind == "top_k":
            membership = certificate.parameters[:self.n_models] / self.n_samples
            joint = certificate.parameters[self.n_models:] / self.n_samples
            risks = np.where(
                relation == 0,
                0.5 * (membership[i] + membership[j] - 2.0 * joint),
                0.5 * (1.0 - relation * (membership[i] - membership[j])),
            )
        elif reusable and certificate.feature_kind == "weak_order":
            weak = certificate.parameters.reshape(self.n_models, self.n_models) / self.n_samples
            risks = np.where(
                relation == 0,
                1.0 - 0.5 * (weak[i, j] + weak[j, i]),
                0.5 * (1.0 - relation * (weak[i, j] - weak[j, i])),
            )
        else:
            true_labels = _sample_report_labels(self.samples(belief), cut_points)
            truth = np.sign(true_labels[:, j] - true_labels[:, i])
            risks = 0.5 * np.mean(np.abs(truth - relation), axis=0)
        return np.clip(risks, 0.0, 1.0)


    def _full_ranking_coverage(self, belief, labels):
        samples = self.samples(belief)
        order = np.argsort(labels)
        left = samples[:, order[:-1]]
        right = samples[:, order[1:]]
        ordered = (left > right) | (
            (left == right) & (order[:-1] < order[1:])
        )
        coverage = float(np.mean(np.all(ordered, axis=1)))
        return coverage


    def summarize(self, belief, cut_points, epsilon, previous=None):
        """Minimize expected Kemeny distance and evaluate the report's posterior credibility."""
        if belief.n_models != self.n_models:
            raise ValueError("belief dimension does not match evaluator")
        # Full-ranking risk is analytic, and credibility checks adjacent inequalities.
        if len(cut_points) == self.n_models and epsilon == 0.0:
            labels, risk = self._full_ranking_report(belief, cut_points)
            coverage = self._full_ranking_coverage(belief, labels)
            return PosteriorSummary(float(np.clip(risk, 0.0, 1.0)), coverage, labels)

        true_labels = _sample_report_labels(self.samples(belief), cut_points)
        q = self.n_models * (self.n_models - 1) // 2
        if len(cut_points) == self.n_models:
            labels, risk = self._full_ranking_report(belief, cut_points)
            certificate = None
        else:
            previous_certificate = previous.report_certificate if previous is not None else None
            labels, risk, certificate = _sample_optimal_report(
                true_labels, cut_points, self.evaluator_id, previous_certificate)
        if len(cut_points) == 2:
            k = cut_points[0]
            overlap = np.sum(true_labels[:, labels == 0] == 0, axis=1)
            d = k - overlap
            distances = d * (self.n_models - d) / q
        else:
            i, j = self.pair_i, self.pair_j
            reported = np.sign(labels[j] - labels[i])
            truth = np.sign(true_labels[:, j] - true_labels[:, i])
            distances = np.abs(truth - reported).sum(axis=1) / (2.0 * q)
        coverage = float(np.mean(distances <= epsilon))
        return PosteriorSummary(float(np.clip(risk, 0.0, 1.0)), coverage, labels, certificate)
