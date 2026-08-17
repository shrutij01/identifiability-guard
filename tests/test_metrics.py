"""Tests for Identifiability Metrics."""

import numpy as np
import pytest

from sklearn import feature_selection, metrics, preprocessing

from scipy.optimize import linear_sum_assignment

try:
    import torch
    from identifiability_guard.metrics.mcc import auction_linear_assignment

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from identifiability_guard.metrics import (
    MCC,
    DCI,
    MIG,
    R2,
    TMEX,
    InfoMEC,
    InfoM,
    InfoE,
    InfoC,
)
from identifiability_guard.metrics.infomec import (
    _compute_nmi_matrix,
    _process_sources,
    EPS,
    sanitize_mi,
    clamp,
)


class TestMCC:
    """Tests for Mean Correlation Coefficient metric."""

    def test_perfect_correlation(self):
        """Test MCC = 1 for perfect correlation."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = Z.copy()  # Perfect identity

        mcc = MCC()
        result = mcc.compute(Z, Z_hat)
        assert result.primary_score > 0.99

    def test_perfect_with_scaling(self):
        """Test MCC = 1 for scaled identity."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = Z * np.array([2.0, -0.5, 3.0])  # Scaled

        mcc = MCC()
        result = mcc.compute(Z, Z_hat)
        assert result.primary_score > 0.99

    def test_perfect_with_permutation(self):
        """Test MCC = 1 for permuted identity."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = Z[:, [2, 0, 1]]  # Permuted

        mcc = MCC()
        result = mcc.compute(Z, Z_hat)
        assert result.primary_score > 0.99

    def test_uncorrelated(self):
        """Test low MCC for uncorrelated variables."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = np.random.randn(1000, 3)  # Independent

        mcc = MCC()
        result = mcc.compute(Z, Z_hat)
        assert result.primary_score < 0.2  # Should be low

    def test_dimension_mismatch_more_codes(self):
        """Test MCC with m > d (overcomplete)."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = np.column_stack([Z, np.random.randn(1000, 2)])

        mcc = MCC()
        result = mcc.compute(Z, Z_hat)
        assert result.primary_score > 0.9  # Should still find good matches

    def test_dimension_mismatch_fewer_codes(self):
        """Test MCC with m < d (undercomplete): legacy divides by m."""
        np.random.seed(42)
        Z = np.random.randn(1000, 5)
        Z_hat = Z[:, :3]  # Only first 3 factors

        mcc = MCC()
        result = mcc.compute(Z, Z_hat)
        # Legacy matched score is high (3 matched pairs are perfect)
        assert result.primary_score > 0.9

    def test_coverage_equals_matched_when_complete(self):
        """When d <= m, coverage score equals matched score."""
        np.random.seed(42)
        Z = np.random.randn(1000, 3)
        Z_hat = Z[:, [2, 0, 1]] * np.array([1.5, -2.0, 0.5])

        result = MCC().compute(Z, Z_hat)
        np.testing.assert_almost_equal(
            result.subscores["mcc"],
            result.subscores["mcc_coverage"],
            decimal=10,
        )

    def test_coverage_penalizes_missing_factors(self):
        """When d > m, coverage < matched because unmatched factors count as zero."""
        np.random.seed(42)
        Z = np.random.randn(1000, 4)
        Z_hat = Z[:, :2]  # only recover 2 of 4 factors

        result = MCC().compute(Z, Z_hat)
        assert result.subscores["mcc"] > 0.9
        assert result.subscores["mcc_coverage"] < result.subscores["mcc"]
        np.testing.assert_almost_equal(
            result.subscores["mcc_coverage"],
            result.subscores["mcc"] * 2 / 4,
            decimal=10,
        )

    def test_coverage_normalization_primary_score(self):
        """normalization='coverage' uses coverage as primary_score."""
        np.random.seed(42)
        Z = np.random.randn(1000, 4)
        Z_hat = Z[:, :2]

        matched = MCC(normalization="matched").compute(Z, Z_hat)
        coverage = MCC(normalization="coverage").compute(Z, Z_hat)
        assert coverage.primary_score < matched.primary_score
        np.testing.assert_almost_equal(
            coverage.primary_score,
            coverage.subscores["mcc_coverage"],
            decimal=10,
        )

    def test_crossfit_mcc(self):
        """Cross-fitted MCC runs and returns a valid score."""
        np.random.seed(42)
        Z = np.random.randn(200, 3)
        Z_hat = Z[:, [2, 0, 1]]

        result = MCC(crossfit=True, seed=42).compute(Z, Z_hat)
        assert 0 <= result.primary_score <= 1
        assert result.metadata["crossfit"] is True

    def test_crossfit_le_legacy(self):
        """Cross-fitted MCC should not exceed legacy in-sample MCC."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z + 0.5 * np.random.randn(500, 3)

        legacy = MCC(seed=42).compute(Z, Z_hat).primary_score
        crossfit = MCC(crossfit=True, seed=42).compute(Z, Z_hat).primary_score
        assert crossfit <= legacy + 1e-6

    def test_crossfit_oos(self):
        """compute_oos with crossfit uses provided train/test split."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z[:, [2, 0, 1]]
        Z_tr, Z_te = Z[:250], Z[250:]
        Zh_tr, Zh_te = Z_hat[:250], Z_hat[250:]

        metric = MCC(crossfit=True, seed=42)
        result = metric.compute_oos(Z_tr, Zh_tr, Z_te, Zh_te)
        assert 0 <= result.primary_score <= 1
        assert result.metadata.get("oos") is True

    def test_compute_from_matrix_rejects_negative(self):
        """compute_from_matrix should reject signed correlations."""
        R = np.array([[0.9, -0.1], [-0.2, 0.8]])
        mcc = MCC()
        with pytest.raises(ValueError, match="absolute correlations"):
            mcc.compute_from_matrix(R)

    def test_too_few_samples(self):
        """Test that too few samples raises error."""
        Z = np.random.randn(1, 3)
        Z_hat = np.random.randn(1, 3)

        mcc = MCC()
        with pytest.raises(ValueError):
            mcc.compute(Z, Z_hat)

    def test_callable(self):
        """Test metric is callable."""
        np.random.seed(42)
        Z = np.random.randn(100, 3)
        Z_hat = Z.copy()

        mcc = MCC()
        result = mcc(Z, Z_hat)
        assert 0 <= result.primary_score <= 1


class TestMCCGoldenValues:
    """Regression test: MCC scores must match values from main.

    Generated via evaluate_dgp_encoder(n_samples=1000, n_factors=5, seed=42)
    using the standard DGP/encoder pipeline. If these drift, either the MCC
    implementation changed (check carefully) or a DGP/encoder changed upstream.
    """

    GOLDEN = {
        "D1_E1": {"mcc_pearson": 1.0, "mcc_spearman": 1.0},
        "D1_E3": {"mcc_pearson": 0.6506353829768304, "mcc_spearman": 0.626043051076277},
        "D1_E4": {"mcc_pearson": 1.0, "mcc_spearman": 1.0},
        "D1_E5": {"mcc_pearson": 0.9999999999999998, "mcc_spearman": 1.0},
        "D1_E9": {
            "mcc_pearson": 0.12266604955661498,
            "mcc_spearman": 0.12211955298882474,
        },
    }

    @pytest.mark.parametrize("case", GOLDEN.keys())
    def test_mcc_matches_golden(self, case):
        from experiments.utils import evaluate_dgp_encoder

        dgp, enc = case.split("_")
        scores = evaluate_dgp_encoder(
            dgp,
            enc,
            metrics_to_compute={"mcc_pearson", "mcc_spearman"},
        )
        for metric, expected in self.GOLDEN[case].items():
            np.testing.assert_allclose(
                scores[metric],
                expected,
                atol=1e-10,
                err_msg=f"{case}/{metric} drifted from golden value",
            )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestAssignmentComparison:
    """Compare auction_linear_assignment (GPU) vs scipy linear_sum_assignment (CPU).

    Both solve max-weight bipartite matching. Auction is O(n²/eps) and GPU-native;
    scipy uses the Jonker-Volgenant algorithm at O(n³) on CPU.
    """

    @staticmethod
    def _scipy_score(C):
        rows, cols = linear_sum_assignment(C, maximize=True)
        return float(C[rows, cols].mean()), rows, cols

    @staticmethod
    def _auction_score(C_np):
        C_t = torch.tensor(C_np, dtype=torch.float32)
        score, assignment, n_iter = auction_linear_assignment(C_t, reduce="mean")
        return float(score), assignment.numpy(), n_iter

    @pytest.mark.parametrize("d", [3, 5, 10, 20, 50])
    def test_square_score_agreement(self, d):
        """On square matrices, both methods should find similar scores."""
        rng = np.random.default_rng(42)
        C = rng.random((d, d)).astype(np.float64)

        scipy_score, _, _ = self._scipy_score(C)
        auction_score, _, _ = self._auction_score(C)

        # Auction is approximate; allow small gap
        assert (
            auction_score <= scipy_score + 1e-6
        ), f"auction ({auction_score:.6f}) > scipy ({scipy_score:.6f})"
        # But should be close
        np.testing.assert_allclose(
            auction_score, scipy_score, atol=0.05, err_msg=f"d={d}: scores diverge"
        )

    def test_permuted_identity_exact(self):
        """Both should perfectly solve a noisy permuted identity."""
        d = 10
        rng = np.random.default_rng(42)
        perm = rng.permutation(d)
        C = 0.01 * rng.random((d, d))
        for i, j in enumerate(perm):
            C[i, j] = 1.0

        scipy_score, scipy_rows, scipy_cols = self._scipy_score(C)
        auction_score, auction_assign, _ = self._auction_score(C)

        np.testing.assert_allclose(auction_score, scipy_score, atol=1e-3)
        # Both should recover the permutation
        assert all(scipy_cols[i] == perm[i] for i in range(d))

    def test_rectangular_overcomplete(self):
        """d < m: compare padded auction vs scipy on overcomplete case."""
        rng = np.random.default_rng(42)
        d, m = 3, 6
        Z = rng.standard_normal((200, d))
        Z_hat = np.column_stack([Z, rng.standard_normal((200, m - d))])

        # scipy path (via numpy MCC)
        from identifiability_guard.metrics.mcc import mean_corr_coef_np

        scipy_score = mean_corr_coef_np(Z, Z_hat, method="pearson")

        # auction path (via pytorch MCC)
        from identifiability_guard.metrics.mcc import mean_corr_coef_pt

        Z_t = torch.tensor(Z, dtype=torch.float32)
        Zh_t = torch.tensor(Z_hat, dtype=torch.float32)
        auction_score = mean_corr_coef_pt(Z_t, Zh_t, method="pearson")

        # Both should find the 3 good matches
        assert scipy_score > 0.9
        assert auction_score > 0.85
        np.testing.assert_allclose(auction_score, scipy_score, atol=0.1)

    def test_rectangular_undercomplete(self):
        """d > m: compare padded auction vs scipy on undercomplete case."""
        rng = np.random.default_rng(42)
        d, m = 6, 3
        Z = rng.standard_normal((200, d))
        Z_hat = Z[:, :m]

        from identifiability_guard.metrics.mcc import (
            mean_corr_coef_np,
            mean_corr_coef_pt,
        )

        scipy_score = mean_corr_coef_np(Z, Z_hat, method="pearson")

        Z_t = torch.tensor(Z, dtype=torch.float32)
        Zh_t = torch.tensor(Z_hat, dtype=torch.float32)
        auction_score = mean_corr_coef_pt(Z_t, Zh_t, method="pearson")

        assert scipy_score > 0.9
        assert auction_score > 0.85
        np.testing.assert_allclose(auction_score, scipy_score, atol=0.1)

    @pytest.mark.parametrize("d", [5, 20, 50, 100])
    def test_timing(self, d):
        """Print wall-clock comparison (not an assertion, just informational)."""
        import time

        rng = np.random.default_rng(42)
        C = rng.random((d, d)).astype(np.float64)
        C_t = torch.tensor(C, dtype=torch.float32)

        # Warmup
        linear_sum_assignment(C, maximize=True)
        auction_linear_assignment(C_t.clone(), reduce="mean")

        n_reps = 50
        t0 = time.perf_counter()
        for _ in range(n_reps):
            linear_sum_assignment(C, maximize=True)
        scipy_ms = (time.perf_counter() - t0) / n_reps * 1000

        t0 = time.perf_counter()
        for _ in range(n_reps):
            auction_linear_assignment(C_t.clone(), reduce="mean")
        auction_ms = (time.perf_counter() - t0) / n_reps * 1000

        print(
            f"\n  d={d:>3d}: scipy={scipy_ms:.3f}ms  auction={auction_ms:.3f}ms  "
            f"ratio={auction_ms/max(scipy_ms, 1e-6):.1f}x"
        )


class TestDCI:
    """Tests for DCI metric."""

    def test_basic_computation(self):
        """Test basic DCI computation."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z.copy()

        dci = DCI()
        scores = dci.compute(Z, Z_hat)

        assert "disentanglement" in scores
        assert "completeness" in scores
        assert "informativeness" in scores

    def test_perfect_disentanglement(self):
        """Test high scores for perfect encoding."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = Z.copy()

        dci = DCI()
        scores = dci.compute(Z, Z_hat)

        # Perfect encoding should have high scores
        assert scores["disentanglement"] > 0.8
        assert scores["completeness"] > 0.8
        assert scores["informativeness"] > 0.9

    def test_entangled_representation(self):
        """Test lower disentanglement for entangled codes."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        # Create entangled representation
        A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        Z_hat = Z @ A.T

        dci = DCI()
        scores = dci.compute(Z, Z_hat)

        # Entangled should have lower disentanglement
        assert scores["disentanglement"] < 0.8

    def test_scores_in_range(self):
        """Test that all scores are in [0, 1]."""
        np.random.seed(42)
        Z = np.random.randn(500, 3)
        Z_hat = np.random.randn(500, 3)

        dci = DCI()
        scores = dci.compute(Z, Z_hat)

        for name, score in scores.items():
            assert 0 <= score <= 1, f"{name} = {score} not in [0, 1]"

    def test_gradient_boosting_method(self):
        """Test DCI with gradient boosting method."""
        np.random.seed(42)
        Z = np.random.randn(200, 3)
        Z_hat = Z.copy()

        dci = DCI(method="gradient_boosting", n_estimators=50)
        scores = dci.compute(Z, Z_hat)

        assert scores["informativeness"] > 0.8

    def test_too_few_samples(self):
        """Test that too few samples raises error."""
        Z = np.random.randn(5, 3)
        Z_hat = np.random.randn(5, 3)

        dci = DCI()
        with pytest.raises(ValueError):
            dci.compute(Z, Z_hat)

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            DCI(method="invalid")

    def test_callable(self):
        """Test metric is callable."""
        np.random.seed(42)
        Z = np.random.randn(200, 3)
        Z_hat = Z.copy()

        dci = DCI()
        scores = dci(Z, Z_hat)
        assert isinstance(scores, dict)


class TestInfoMECVectorization:
    """Tests for InfoMEC NMI vectorization correctness."""

    def test_vectorisation_matches_loop(self):
        rng = np.random.default_rng(0)
        n = 200
        num_sources = 4
        num_latents = 6
        sources = rng.normal(size=(n, num_sources))
        latents = rng.normal(size=(n, num_latents))

        n_neighbors = 5
        num_bins = 10
        random_state = 0

        nmi_vec = _compute_nmi_matrix(
            sources=sources,
            latents=latents,
            discrete_latents=False,
            n_neighbors=n_neighbors,
            num_bins=num_bins,
            random_state=random_state,
        )

        processed_sources = _process_sources(sources, num_bins=num_bins)
        processed_latents = preprocessing.StandardScaler().fit_transform(latents)

        nmi_loop = np.empty((num_sources, num_latents))
        for i in range(num_sources):
            entropy_i = metrics.mutual_info_score(
                processed_sources[:, i], processed_sources[:, i]
            )
            if entropy_i < EPS:
                nmi_loop[i, :] = 0.0
                continue
            for j in range(num_latents):
                mi_ij = feature_selection.mutual_info_classif(
                    processed_latents[:, j][:, None],
                    processed_sources[:, i],
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                )[0]
                mi_ij = sanitize_mi(mi_ij)
                nmi_loop[i, j] = clamp(mi_ij / entropy_i)

        np.testing.assert_allclose(nmi_vec, nmi_loop, rtol=1e-2, atol=2e-4)


def _make_data(n=200, d=5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d))
    Z_hat = Z + noise * rng.standard_normal((n, d))
    return Z, Z_hat


class TestReproducibility:
    """All metrics should be deterministic when seeded."""

    # -- MCC (RDC uses random projections) --

    def test_mcc_rdc_seeded(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="rdc", seed=42).compute(Z, Z_hat)
        r2 = MCC(method="rdc", seed=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_mcc_rdc_different_seeds(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="rdc", seed=1).compute(Z, Z_hat)
        r2 = MCC(method="rdc", seed=2).compute(Z, Z_hat)
        assert r1.primary_score != r2.primary_score

    def test_mcc_rdc_unseeded_varies(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="rdc").compute(Z, Z_hat)
        r2 = MCC(method="rdc").compute(Z, Z_hat)
        assert r1.primary_score != r2.primary_score

    def test_mcc_pearson_deterministic(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="pearson").compute(Z, Z_hat)
        r2 = MCC(method="pearson").compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_mcc_spearman_deterministic(self):
        Z, Z_hat = _make_data()
        r1 = MCC(method="spearman").compute(Z, Z_hat)
        r2 = MCC(method="spearman").compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    # -- DCI (train/test split uses random_state) --

    def test_dci_seeded(self):
        Z, Z_hat = _make_data()
        r1 = DCI(random_state=42).compute(Z, Z_hat)
        r2 = DCI(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_dci_different_seeds(self):
        Z, Z_hat = _make_data()
        r1 = DCI(random_state=1).compute(Z, Z_hat)
        r2 = DCI(random_state=2).compute(Z, Z_hat)
        assert r1.primary_score != r2.primary_score

    # -- InfoMEC (MI estimation + logistic regression use randomness) --

    def test_infomec_seeded(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoMEC(random_state=42).compute(Z, Z_hat)
        r2 = InfoMEC(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_infomec_different_seeds(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoMEC(random_state=1).compute(Z, Z_hat)
        r2 = InfoMEC(random_state=2).compute(Z, Z_hat)
        assert r1.primary_score != r2.primary_score

    def test_infom_seeded(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoM(random_state=42).compute(Z, Z_hat)
        r2 = InfoM(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_infoe_seeded(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoE(random_state=42).compute(Z, Z_hat)
        r2 = InfoE(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_infoc_seeded(self):
        Z, Z_hat = _make_data(n=300, d=3)
        r1 = InfoC(random_state=42).compute(Z, Z_hat)
        r2 = InfoC(random_state=42).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    # -- TMEX (PCM test uses random splits) --

    def test_tmex_seeded(self):
        Z, Z_hat = _make_data(n=200, d=3)
        r1 = TMEX(seed=42, rep=3).compute(Z, Z_hat)
        r2 = TMEX(seed=42, rep=3).compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    def test_tmex_different_seeds(self):
        Z, Z_hat = _make_data(n=200, d=3)
        r1 = TMEX(seed=1, rep=3).compute(Z, Z_hat)
        r2 = TMEX(seed=2, rep=3).compute(Z, Z_hat)
        # TMEX returns binary (0/1) scores so different seeds may still match;
        # we only check that the call succeeds and returns valid scores
        assert 0.0 <= r1.primary_score <= 1.0
        assert 0.0 <= r2.primary_score <= 1.0

    # -- MIG (deterministic: histogram binning + discrete MI) --

    def test_mig_deterministic(self):
        Z, Z_hat = _make_data()
        r1 = MIG().compute(Z, Z_hat)
        r2 = MIG().compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score

    # -- R2 (deterministic: least squares) --

    def test_r2_deterministic(self):
        Z, Z_hat = _make_data()
        r1 = R2().compute(Z, Z_hat)
        r2 = R2().compute(Z, Z_hat)
        assert r1.primary_score == r2.primary_score
