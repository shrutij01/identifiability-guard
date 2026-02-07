"""
Demo of the unified metrics API.

Shows how to use the MetricRegistry to compute identifiability metrics
with different DGPs (data generating processes) and encoders.

Usage:
    # Interactive mode - prompts for DGP and encoder selection
    python examples/metrics_demo.py

    # Non-interactive mode - specify DGP and encoder
    python examples/metrics_demo.py <dgp> <encoder>

    # Compact mode - just print the three main metrics (DCI, MCC, R²)
    python examples/metrics_demo.py <dgp> <encoder> --compact

    # Examples:
    python examples/metrics_demo.py d1 e1             # Full demo
    python examples/metrics_demo.py d2 e3 --compact   # Compact output
    python examples/metrics_demo.py d4 e5             # Full demo

Available DGPs:
    d1: Independent, non-redundant factors
    d2: Correlated, non-redundant factors
    d3: Single-factor redundant
    d4: Multi-factor redundant

Available Encoders:
    e1: Elementwise Linear
    e2: Elementwise Nonlinear
    e3: Linearly Entangled
    e4: Undercomplete Linear
    e5: Overcomplete Linear
    e6: Overcomplete Multicodes
"""

import sys
import numpy as np
from src.metrics import MetricRegistry
from src.dgp import D1Independent, D2Correlated, D3SingleRedundant, D4MultiRedundant
from src.encoders import (
    E1ElementwiseLinear,
    E2ElementwiseNonlinear,
    E3LinearlyEntangled,
    E4UndercompleteLinear,
    E5OvercompleteLinear,
    E6OvercompleteMulticodes,
    E7OvercompleteEntangled,
    E8OvercompleteDisjoint,
    E9RandomGaussian,
    E10RandomUniform,
)


# Available DGPs and Encoders
DGP_OPTIONS = {
    "d1": ("D1: Independent", D1Independent),
    "d2": ("D2: Correlated", D2Correlated),
    "d3": ("D3: Single Redundant", D3SingleRedundant),
    "d4": ("D4: Multi Redundant", D4MultiRedundant),
}

ENCODER_OPTIONS = {
    "e1": ("E1: Elementwise Linear", E1ElementwiseLinear),
    "e2": ("E2: Elementwise Nonlinear", E2ElementwiseNonlinear),
    "e3": ("E3: Linearly Entangled", E3LinearlyEntangled),
    "e4": ("E4: Undercomplete Linear", E4UndercompleteLinear),
    "e5": ("E5: Overcomplete Linear", E5OvercompleteLinear),
    "e6": ("E6: Overcomplete Multicodes", E6OvercompleteMulticodes),
    "e7": ("E7: Overcomplete Entangled", E7OvercompleteEntangled),
    "e8": ("E8: Overcomplete Disjoint", E8OvercompleteDisjoint),
    "e9": ("E9: Random Gaussian", E9RandomGaussian),
    "e10": ("E10: Random Uniform", E10RandomUniform),
}


def select_option(options, prompt):
    """Interactive selection from options."""
    print(f"\n{prompt}")
    for key, (description, _) in options.items():
        print(f"  {key}: {description}")

    while True:
        choice = input("Select: ").strip().lower()
        if choice in options:
            return choice
        print(f"Invalid choice. Please select from: {list(options.keys())}")


def main(dgp_choice=None, encoder_choice=None, compact=False):
    if not compact:
        print("=" * 70)
        print("Identifiability Metrics - Unified API Demo")
        print("=" * 70)

    # Interactive or default selection
    if dgp_choice is None or encoder_choice is None:
        print("\nRunning interactive mode...")
        dgp_choice = select_option(DGP_OPTIONS, "Select Data Generating Process:")
        encoder_choice = select_option(ENCODER_OPTIONS, "Select Encoder:")

    # Get selected DGP and encoder
    dgp_name, DGPClass = DGP_OPTIONS[dgp_choice]
    encoder_name, EncoderClass = ENCODER_OPTIONS[encoder_choice]

    if not compact:
        print(f"\n{'='*70}")
        print(f"Configuration:")
        print(f"  DGP:     {dgp_name}")
        print(f"  Encoder: {encoder_name}")
        print(f"{'='*70}")

    # Generate data
    n_samples = 1000 if compact else 100
    n_factors = 4
    seed = 42

    # Initialize DGP
    dgp = DGPClass(d=n_factors, seed=seed)
    if not compact:
        print(f"\nGenerating {n_samples} samples from {dgp.name}...")
    Z = dgp.sample(n_samples)

    # Initialize encoder
    encoder = EncoderClass(d=n_factors, seed=seed)
    if not compact:
        print(f"Encoding with {encoder.name}...")
    Z_hat = encoder.encode(Z)

    # Create registry and register default metrics
    registry = MetricRegistry()
    registry.register_defaults()

    # Compact mode: just print the three main metrics
    if compact:
        print("=" * 70)
        print(f"{dgp_name} × {encoder_name}")
        print(f"Samples: {n_samples}, Factors: {n_factors}, Shape: Z{Z.shape} → Z^{Z_hat.shape}")
        print("=" * 70)

        # Compute metrics
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            dci = registry.create("dci").compute(Z, Z_hat)
            mcc_p = registry.create("mcc_pearson").compute(Z, Z_hat)
            mcc_s = registry.create("mcc_spearman").compute(Z, Z_hat)
            r2 = registry.create("r2").compute(Z, Z_hat)
            mig = registry.create("mig").compute(Z, Z_hat)
            tmex = registry.create("tmex").compute(Z, Z_hat)
            infom = registry.create("infom").compute(Z, Z_hat)
            infoe = registry.create("infoe").compute(Z, Z_hat)
            infoc = registry.create("infoc").compute(Z, Z_hat)

        print()
        print("DCI (Disentanglement, Completeness, Informativeness)")
        print(f"  Disentanglement:      {dci.subscores['disentanglement']:.3f}")
        print(f"  Completeness:         {dci.subscores['completeness']:.3f}")
        print(f"  Informativeness:      {dci.subscores['informativeness_test']:.3f}")
        print()
        print("MCC (Mean Correlation Coefficient)")
        print(f"  Pearson:              {mcc_p.primary_score:.3f}")
        print(f"  Spearman:             {mcc_s.primary_score:.3f}")
        print()
        print("R² Score")
        print(f"  R² (optimal):         {r2.primary_score:.3f}")
        print()
        print("MIG (Mutual Information Gap)")
        print(f"  MIG:                  {mig.primary_score:.3f}")
        print()
        print("T-MEX (Testing for Measurement Exchangeability)")
        print(f"  T-MEX:                {tmex.primary_score:.3f}")
        print()
        print("InfoMEC (Information-theoretic Modularity, Explicitness, Compactness)")
        print(f"  InfoM (Modularity):   {infom.primary_score:.3f}")
        print(f"  InfoE (Explicitness): {infoe.primary_score:.3f}")
        print(f"  InfoC (Compactness):  {infoc.primary_score:.3f}")
        print()
        print("=" * 70)
        return

    print(f"\nData shapes:")
    print(f"  Ground truth Z:    {Z.shape}")
    print(f"  Learned codes Z^:  {Z_hat.shape}")

    print(f"\nRegistered metrics: {registry.list_metrics()}")

    print("\n" + "=" * 70)
    print("1. Computing Individual Metrics")
    print("=" * 70)

    # Example 1: Use individual metric
    dci = registry.create("dci")
    result = dci.compute(Z, Z_hat)

    print(f"\nDCI Metric:")
    print(f"  Primary score: {result.primary_score:.3f}")
    print(f"  Subscores:")
    for name, value in result.subscores.items():
        print(f"    {name}: {value:.3f}")

    # Example 2: MCC with different correlation methods
    for method in ["pearson", "spearman"]:
        mcc = registry.create(f"mcc_{method}")
        result = mcc.compute(Z, Z_hat)
        print(f"\nMCC ({method}): {result.primary_score:.3f}")

    print("\n" + "=" * 70)
    print("2. Computing All Metrics at Once")
    print("=" * 70)

    # Compute all metrics
    results = registry.compute_all(Z, Z_hat)

    print(f"\nResults for all metrics:")
    for name in sorted(results.keys()):
        score = results[name].primary_score
        print(f"  {name:20s}: {score:.3f}")

    print("\n" + "=" * 70)
    print("3. Computing from Precomputed Matrix")
    print("=" * 70)

    # Example: Compute correlation matrix once, reuse for MCC
    corr_matrix = np.abs(np.corrcoef(Z, Z_hat, rowvar=False)[:n_factors, n_factors:])

    print(f"\nPrecomputed correlation matrix shape: {corr_matrix.shape}")

    mcc = registry.create("mcc_pearson")
    result_from_matrix = mcc.compute_from_matrix(corr_matrix)
    result_from_samples = mcc.compute(Z, Z_hat)

    print(f"MCC from matrix:  {result_from_matrix.primary_score:.3f}")
    print(f"MCC from samples: {result_from_samples.primary_score:.3f}")
    print(f"Difference:       {abs(result_from_matrix.primary_score - result_from_samples.primary_score):.6f}")

    print("\n" + "=" * 70)
    print("4. Error Handling")
    print("=" * 70)

    try:
        # This should fail - R² can't be computed from matrix
        r2 = registry.create("r2")
        r2.compute_from_matrix(corr_matrix)
    except NotImplementedError as e:
        print(f"\nExpected error for R² from matrix:")
        print(f"  {e}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Parse command line arguments
    compact_mode = "--compact" in sys.argv
    if compact_mode:
        sys.argv.remove("--compact")

    if len(sys.argv) == 3:
        # Non-interactive mode with arguments
        dgp_arg = sys.argv[1].lower()
        encoder_arg = sys.argv[2].lower()

        if dgp_arg not in DGP_OPTIONS:
            print(f"Error: Invalid DGP '{dgp_arg}'. Choose from: {list(DGP_OPTIONS.keys())}")
            sys.exit(1)
        if encoder_arg not in ENCODER_OPTIONS:
            print(f"Error: Invalid encoder '{encoder_arg}'. Choose from: {list(ENCODER_OPTIONS.keys())}")
            sys.exit(1)

        main(dgp_choice=dgp_arg, encoder_choice=encoder_arg, compact=compact_mode)
    elif len(sys.argv) == 1:
        # Interactive mode
        main(compact=compact_mode)
    else:
        print("Usage:")
        print("  python metrics_demo.py                          # Interactive mode")
        print("  python metrics_demo.py <dgp> <encoder>          # Non-interactive mode")
        print("  python metrics_demo.py <dgp> <encoder> --compact  # Compact output")
        print()
        print("DGP options:", list(DGP_OPTIONS.keys()))
        print("Encoder options:", list(ENCODER_OPTIONS.keys()))
        print()
        print("Examples:")
        print("  python metrics_demo.py d1 e1")
        print("  python metrics_demo.py d2 e3 --compact")
        sys.exit(1)
