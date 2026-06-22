\section{Metrics as Measurement Instruments}\label{sec:results}

We study the structural sensitivity of identifiability metrics through controlled synthetic experiments. In each experiment, we sample ground-truth factors $\rvz \in \sR^d$ according to a DGP type (D1--D4) and construct representations $\hat{\rvz} = T(\rvz)$ via a transformation matching the encoder type (E1--E10); no learner is involved. This design isolates metric misspecification from optimisation artefacts: every failure we observe is a property of the metric, not of training. Unless otherwise noted, we report results for $n = 1000$ samples, $d = 5$ ground-truth factors, and average over 5 seeds; confidence bands show 95\% intervals. Full experimental details and parameter definitions are in \cref{sec:app_experiments}.

\paragraph{Metrics evaluated.} We evaluate four representative metrics spanning three families: \mccP{} and \mccS{} (correlation-based), \dciD{} (regression-based), and \rsq{} (variance-explained). Results for the full set of 12 metrics (including MIG, InfoMEC, T-MEX, MCC-RDC) are reported in the appendix (\cref{fig:app_full_metrics}); we focus on these four because they are the most widely used and their failures are representative of their respective families.

We first ask: \emph{do metric scores remain stable when the encoder perfectly recovers each factor (E1), but the DGP varies from independent to correlated to functionally redundant?} Any metric faithful to the equivalence class should return $\approx 1$ across D1--D4 under E1, since the encoder--factor relationship is identical in all cases. Under D1~+~E1, factors are independent and each code is a rescaled copy of one factor; under D3~+~E1, one factor is a deterministic function of another (e.g.\ $z_2 = z_1^3$), yet every code still maps to exactly one factor via rescaling. The identifiability status is unchanged---only the joint distribution $p(\rvz)$ differs.

We find that \mccP{}, \mccS{}, and \rsq{} pass this test, while \dciD{} exhibits a systematic dip under D3, particularly at small $d$ (\cref{fig:app_dgp_invariance}; the dip diminishes as $d$ grows from 5 to 20 but does not vanish). The dip arises because the redundant factor creates collinearity in the regression probe, inflating the importance mass assigned to the dependent factor and reducing the disentanglement score. This is not a finite-sample artefact: it persists as $n \to \infty$ (\cref{fig:app_dgp_invariance_samplesize}).

A second baseline tests sensitivity to \emph{encoder} nonlinearity rather than DGP structure. Under D1~+~E2, each code is a smooth invertible function of exactly one factor (e.g.\ $\hat{z}_j = \tanh(\alpha \cdot z_{\pi(j)})$), varying $\alpha$ from 0 (linear, reducing to E1) to 1 (fully nonlinear). Since identifiability up to elementwise nonlinear maps ($\mathcal{G}_{\text{nl}}$) is preserved for all $\alpha$, a metric targeting this equivalence class should remain flat. We find that \mccS{} and \dciD{} are indeed flat, while \mccP{} and \rsq{} degrade monotonically with $\alpha$ (\cref{fig:app_nonlinearity_sensitivity})---a false negative arising from the linearity assumption in Pearson correlation and linear regression respectively.


%% ============================================================
\subsection{Correlation sign and magnitude create spurious score variation}\label{sec:correlation}
%% ============================================================

We next study how latent-factor correlation (D2) interacts with metric scores under encoders E1 (perfectly disentangled) and E3 (linearly entangled). The controlled parameter is the pairwise correlation $\rho \in (-1, 1)$, set uniformly across all off-diagonal entries of $\Sigma$. The encoder is held fixed; only $\rho$ varies.

\paragraph{Setup: D2 + E1.} Consider $d$ ground-truth factors with
\[
  \rvz \sim \mathcal{N}(\mathbf{0}, \Sigma), \qquad \Sigma_{ii} = 1, \quad \Sigma_{ij} = \rho \text{ for } i \neq j,
\]
and the elementwise encoder $\hat{z}_j = s_j z_j$ with $s_j > 0$.  The encoder is identical for $\rho = +0.5$ and $\rho = -0.5$: only the sign of the off-diagonal entries of $\Sigma$ changes.  A faithful metric should assign the same score to both.

\paragraph{Setup: D2 + E3.} Same latent distribution, but now the encoder is a full-rank linear map:
\[
  \hat{\rvz} = A \rvz + \mathbf{b}, \qquad A = U \, \text{diag}(\text{linspace}(1, \kappa^{-1}, d)) \, V^\top,
\]
where $U, V$ are random orthogonal matrices and $\kappa \geq 1$ controls the condition number (degree of entanglement). At $\kappa = 1$, $A$ is orthogonal; as $\kappa \to \infty$, the mixing becomes ill-conditioned.

\begin{figure}[t]
    \centering
    % PLACEHOLDER: Image 3 — Sign asymmetry across dimensionalities
    \includegraphics[width=\linewidth]{figures/fig_sign_asymmetry.pdf}
    \caption{\textbf{Metrics are sensitive to the sign of latent correlation despite identical encoder quality.} Left column: D2~+~E1 (perfectly disentangled). All metrics should be flat at $\approx 1$; \dciD{} drops asymmetrically for $\rho < 0$, especially at $d = 2$.  Right column: D2~+~E3 (linearly entangled).  Scores should reflect entanglement but not correlation sign; instead, all metrics vary with both $\rho$ and its sign.  Rows: $d \in \{2, 5, 10\}$.  See \cref{fig:app_sign_asymmetry_allmetrics} for all 12 metrics.}
    \label{fig:sign_asymmetry}
\end{figure}

\cref{fig:sign_asymmetry} reveals a striking violation. Under E1 (left), \dciD{} assigns substantially lower scores for $\rho < 0$ than for $\rho > 0$ at $d = 2$, despite identical identifiability status.  The asymmetry diminishes with increasing $d$ but remains detectable at $d = 10$.  Under E3 (right), the interaction compounds: scores vary jointly with $\rho$ and $\kappa$, making it impossible to attribute a low score to either correlation or entanglement from the metric value alone.

% MATH PLACEHOLDER:
% \begin{proposition}[Metric dependence on correlation under D2 + E1]\label{prop:corr_dependence}
% Let $\rvz \sim \mathcal{N}(\mathbf{0}, \Sigma)$ with $\Sigma_{ij} = \rho$ for $i \neq j$, and $\hat{z}_j = s_j z_j$.  Then:
% \begin{enumerate}
%   \item $\mccP = 1$ for all $\rho$, $d$, and $n \to \infty$.
%   \item $\mccS = 1$ for all $\rho$, $d$, and $n \to \infty$ (rank correlation is invariant to marginal transforms).
%   \item $\rsq = 1$ for all $\rho$, since $z_j$ is a linear function of $\hat{z}_j$.
%   \item $\dciD = g(\rho, d)$ where $g$ depends on the regularisation path of the Lasso used in DCI.  Specifically, for $d = 2$:
%   \[
%     \dciD(\rho) \neq \dciD(-\rho) \quad \text{when the Lasso regularisation interacts with } \text{sign}(\Sigma_{12}).
%   \]
%   [Derive the explicit form of $g$ for $d=2$ Gaussian case.]
% \end{enumerate}
% \end{proposition}

% MATH PLACEHOLDER:
% \begin{proposition}[\rsq{} under D2 + E3]\label{prop:r2_affine}
% Let $\hat{\rvz} = A \rvz + \mathbf{b}$ with $A$ invertible.  Then $\rsq = 1$ for all $A$, $\Sigma$, and $d$.
%
% \emph{Proof sketch.} Each $z_j$ is a linear function of $\hat{\rvz}$ (via $A^{-1}$), so the linear regression achieves perfect prediction.  \qed
%
% \emph{Consequence:} \rsq{} cannot distinguish E1 from E3---it is structurally insensitive to linear entanglement.
% \end{proposition}

\paragraph{Separating correlation from entanglement.} To disentangle these two effects, we vary $\rho$ and $\kappa$ jointly under D2~+~E3.

\begin{figure}[t]
    \centering
    % PLACEHOLDER: Image 4 — Correlation vs Entanglement heatmaps
    \includegraphics[width=\linewidth]{figures/fig_corr_vs_entanglement.pdf}
    \caption{\textbf{Correlation magnitude, not entanglement strength, dominates metric scores.}  Heatmaps of metric scores as a function of correlation $\rho$ (rows) and condition number $\kappa$ (columns) under D2~+~E3 ($d{=}2$, $n{=}100$).  An ideal metric would vary only along $\kappa$ (columns: increasing entanglement should lower scores) and be constant across $\rho$ (rows).  \dciD{} and \mccP{} vary primarily along $\rho$.  \rsq{} is constant everywhere---a false positive for entangled encoders (\cref{prop:r2_affine}).  \mccS{} shows partial dependence on both.  See \cref{fig:app_corr_vs_ent_d5,fig:app_corr_vs_ent_d10} for $d \in \{5, 10\}$.}
    \label{fig:corr_vs_ent}
\end{figure}

\cref{fig:corr_vs_ent} is the central diagnostic for the ``correlation leaks into scores'' cell of \cref{fig:phase}. The ideal heatmap would show variation only along columns ($\kappa$); instead, \dciD{} and \mccP{} are governed by rows ($\rho$).  \rsq{} is flat across the entire grid---consistent with \cref{prop:r2_affine}, but this means it produces false positives for E3 (it scores identically whether the encoder is axis-aligned or fully entangled).  Negative $\rho$ yields systematically different scores from positive $\rho$ at equal $|\rho|$, confirming the sign asymmetry observed in \cref{fig:sign_asymmetry}.

\paragraph{Takeaway.} Under correlated factors, \dciD{} and \mccP{} conflate correlation magnitude (and sign) with entanglement; \rsq{} ignores entanglement entirely.  No metric in this set cleanly separates the two effects.


%% ============================================================
\subsection{Dimension mismatch: undercomplete encoders and factor redundancy}\label{sec:undercomplete}
%% ============================================================

When the encoder outputs fewer dimensions than the number of ground-truth factors ($m < d$, encoder E4), the representation can at most recover a subset $S \subseteq \{1, \ldots, d\}$ with $|S| = m$.  We construct E4 by selecting $m$ factors and applying elementwise rescaling, so the retained factors are \emph{perfectly} identified.  The central question is: \emph{how do metrics score a representation that perfectly identifies $m$ out of $d$ factors, and can they distinguish lossy from lossless omission?}

\paragraph{Setup: D1 + E4 (independent factors, subset recovery).}
\[
  z_j \sim \mathcal{N}(0,1) \text{ independently}, \qquad \hat{z}_j = s_j z_j \text{ for } j \in S, \quad |S| = m < d.
\]
All $d$ factors are independently informative, so omitting any factor is genuinely lossy.

\paragraph{Setup: D3 + E4 (redundant factor, subset recovery).}
\[
  z_1 \sim \mathcal{N}(0,1), \quad z_2 = z_1^3, \quad z_3, \ldots, z_d \sim \mathcal{N}(0,1) \text{ independently},
\]
\[
  \hat{z}_j = s_j z_j \text{ for } j \in S, \quad |S| = m, \quad z_2 \notin S.
\]
Here $z_2$ is a deterministic function of $z_1$, so $\deff = d - 1$.  Dropping $z_2$ is \emph{lossless}: $|S| = d - 1 = \deff$ and the encoder captures all independently varying information.

\begin{figure}[t]
    \centering
    % PLACEHOLDER: Image 7 — Dropping factors across D1–D4
    \includegraphics[width=\linewidth]{figures/fig_dropping_across_dgp.pdf}
    \caption{\textbf{No metric distinguishes lossless compression from lossy omission.}  Metric scores as retained factors $m$ decrease from $d{=}10$.  Under D1 (left), every dropped factor is independently informative---declining scores are correct for \dciD{} and \rsq{}.  Under D3 (third panel), the first dropped factor is redundant ($m = \deff$), so scores should remain at 1.0 before declining---yet \dciD{} and \rsq{} decline immediately.  \mccP{}/\mccS{} report 1.0 everywhere: correct for D3 but a false positive for D1.  Dashed line: $m = d$.  See \cref{fig:app_dropping_detail,fig:app_redundancy_vs_compression} for metric-wise breakdowns and the redundancy $\times$ compression heatmap.}
    \label{fig:dropping_across_dgp}
\end{figure}

\cref{fig:dropping_across_dgp} reveals a fundamental design split.  Correlation-based metrics (\mccP{}, \mccS{}) perform optimal one-to-one matching and evaluate only matched pairs: even $m = 1$ scores 1.0.  Regression-based metrics (\dciD{}, \rsq{}) aggregate over all $d$ factors, so unmatched factors contribute zero, producing a monotone decline.

The split becomes consequential under D3/D4.  Under D3 (third panel), the first factor dropped ($z_2 = z_1^3$) carries no independent information, yet \dciD{} and \rsq{} still decline---they treat $|S| < d$ uniformly, whether the omitted factor is redundant or independently informative.  Conversely, \mccP{}/\mccS{} report 1.0 in both D1 and D3: correct for D3 (lossless compression) but a false positive for D1 (lossy omission).

% MATH PLACEHOLDER:
% \begin{proposition}[Undercomplete metric behaviour]\label{prop:undercomplete}
% Under E4 with $|S| = m$ perfectly identified factors out of $d$ total, as $n \to \infty$:
% \begin{enumerate}
%   \item \mccP{} $= 1$ for all $1 \leq m \leq d$, since the Hungarian matching finds a perfect $|\rho| = 1$ pair for each retained factor and the score averages over the $m$ matched pairs.
%   \item \mccS{} $= 1$ for all $1 \leq m \leq d$, by the same argument with Spearman rank correlation.
%   \item Under D1 (independent factors): $\rsq = m/d$, since each retained factor contributes $1/d$ to the average explained variance and each omitted factor contributes $0$.
%   \item Under D1: $\dciD = f(m/d)$ where $f$ is increasing with $f(0) = 0$ and $f(1) = 1$.  [Derive exact form for Lasso-based DCI.]
%   \item No metric conditions on $\deff$: the score under D3 + E4 with $|S| = \deff$ is identical to the score under D1 + E4 with $|S| = \deff < d$, despite the former being lossless.
% \end{enumerate}
% \end{proposition}

\paragraph{Takeaway.} The undercomplete regime exposes a fundamental gap: no current metric conditions on $\deff$.  Correlation-based metrics ignore partial recovery entirely; regression-based metrics penalise it uniformly.  Neither can distinguish lossless compression from genuine information loss.


%% ============================================================
\subsection{Overcomplete encoders: duplication, distribution, and spurious reward}\label{sec:overcomplete}
%% ============================================================

When $m > d$, the encoder outputs more codes than there are factors.  We compare four overcomplete geometries against matched-dimension baselines across all DGP types.

\paragraph{Setup: E3 (matched, entangled) vs E7 (overcomplete, entangled).}  Under E3 ($m = d$), the encoder is a full-rank linear map $\hat{\rvz} = A\rvz$ that mixes all factors within each code.  Under E7 ($m > d$), we use a fat matrix $\tilde{A} \in \sR^{m \times d}$ with $m > d$: codes still linearly mix factors, but the representation is overcomplete.  In both cases, factors are recoverable only via a linear readout that unmixes the superposition (\cref{sec:taxonomy}).  A faithful metric should score E7 no higher than E3, since E7 adds dimensionality without improving disentanglement.

\begin{figure}[t]
    \centering
    % PLACEHOLDER: Image 10 — E3 vs E7 across DGPs
    \includegraphics[width=\linewidth]{figures/fig_e3_vs_e7.pdf}
    \caption{\textbf{Metrics spuriously reward overcomplete entanglement over matched entanglement.}  \mccP{} and \mccS{} assign \emph{higher} scores to E7 (overcomplete, entangled) than to E3 (matched, entangled) across all DGP types, despite E7 being no better disentangled.  \rsq{} assigns $\approx 1$ to both (\cref{prop:r2_affine}: insensitive to linear entanglement).  \dciD{} is the only metric that scores E7 $\leq$ E3 under D1 and D2. $d{=}5$.  See \cref{fig:app_encoding_type_bars} for comparisons across E1, E5, E6, E8.}
    \label{fig:e3_vs_e7}
\end{figure}

\cref{fig:e3_vs_e7} reveals that \mccP{} and \mccS{} score E7 \emph{higher} than E3 across all DGP types.  The mechanism is combinatorial: MCC solves an optimal matching over an $m \times d$ correlation matrix.  When $m > d$, there are $\binom{m}{d}$ possible matchings; the maximum over a larger set is stochastically larger, inflating the score even when the additional codes are entangled.

% MATH PLACEHOLDER:
% \begin{proposition}[MCC inflation under overcompleteness]\label{prop:mcc_overcomplete}
% Let $C \in \sR^{m \times d}$ be the absolute correlation matrix between $\hat{\rvz}$ and $\rvz$.  MCC solves:
% \[
%   \mcc = \frac{1}{d} \max_{\sigma \in \mathcal{I}(m,d)} \sum_{j=1}^{d} C_{\sigma(j), j},
% \]
% where $\mathcal{I}(m,d)$ is the set of injections from $\{1,\ldots,d\}$ to $\{1,\ldots,m\}$.  For $m > d$, $|\mathcal{I}(m,d)| = m!/(m-d)! > d!$, and $\E[\mcc]$ is non-decreasing in $m$ even when additional codes carry no new per-factor information.
%
% [Derive explicit bound or expectation for the Gaussian case where $\tilde{A}$ is random.]
% \end{proposition}

Under D3 and D4, the effect compounds with functional redundancy: the redundant factor provides additional codes that happen to correlate with ground-truth factors through the deterministic constraint, further inflating the matching.  For the disjoint overcomplete encoders (E5, E6, E8), which preserve a one-factor-per-code structure despite $m > d$, the picture is more benign: \mccS{} scores remain at 1.0, while \mccP{} and \rsq{} show modest degradation under E6 (nonlinear duplication) and E8 (distributed encoding) (\cref{fig:app_encoding_type_bars}).

\paragraph{Takeaway.}  Overcompleteness interacts with the Hungarian matching in MCC to produce systematic false positives for entangled representations.  Metrics that evaluate per-code rather than per-factor are particularly vulnerable.


%% ============================================================
\subsection{Metric scores as a function of the dimensionality ratio $m/d$}\label{sec:ratio}
%% ============================================================

The preceding sections vary one axis at a time.  We now ask a unifying question: \emph{how does each metric respond to the structural ratio $m/d$ when all retained or duplicated factors are perfectly identified?}  We sweep $m \in \{1, \ldots, d\}$ under D1~+~E4 (undercomplete, $m < d$) and extend beyond $m = d$ via E5 (overcomplete duplication).

\paragraph{Setup.} Fix $d = 10$.  For $m \leq d$: select $m$ factors uniformly, apply elementwise rescaling (E4).  For $m > d$: include all $d$ factors plus $m - d$ linear duplicates of randomly chosen factors (E5).  In both regimes, every factor present in $\hat{\rvz}$ is a rescaled copy of exactly one ground-truth factor.

\begin{figure}[t]
    \centering
    % PLACEHOLDER: Image 15 — Metric score vs m/d, all metrics
    \includegraphics[width=\linewidth]{figures/fig_metric_vs_ratio.pdf}
    \caption{\textbf{Three metric families respond differently to the dimensionality ratio.} Under D1, $d{=}10$ fixed, varying $m$.  \emph{Ratio-invariant} metrics (\mccP{}, \mccS{}, T-MEX, MCC-RDC) are flat at 1.0: they evaluate only matched codes.  \emph{Ratio-linear} metrics (\dciD{}, \rsq{}, MIG) scale proportionally to $m/d$.  \emph{Ratio-sensitive nonlinear} metrics (DCI-C, InfoC, InfoE) show complex dependence.  No family is correct across the full range: ratio-invariant metrics miss lossy omission; ratio-linear metrics cannot credit lossless compression.  See \cref{fig:app_ratio_collapse_sweeps} for evidence that $m/d$---not $m$ or $d$ individually---governs the score.}
    \label{fig:ratio}
\end{figure}

\cref{fig:ratio} provides a unified view of the dimension-mismatch landscape.  The three families reflect distinct implicit assumptions.  Ratio-invariant metrics assume evaluation should be per-matched-code (agnostic to coverage of the ground-truth).  Ratio-linear metrics assume evaluation should be per-ground-truth-factor (penalising incomplete coverage uniformly).  The nonlinear group reflects more complex internal normalisations (e.g.\ entropy-based).

% MATH PLACEHOLDER:
% \begin{proposition}[Ratio dependence]\label{prop:ratio}
% Under D1 + E4, with $m$ perfectly identified factors out of $d$, as $n \to \infty$:
% \begin{enumerate}
%   \item $\mccP(m, d) = 1$ for all $m \geq 1$.
%   \item $\rsq(m, d) = m/d$.
%   \item $\dciD(m, d) = f(m/d)$ where $f$ is increasing, concave, with $f(0) = 0$, $f(1) = 1$.  [Derive $f$ for the Lasso case.]
% \end{enumerate}
% Moreover, for all three metrics, the score depends on $(m, d)$ only through the ratio $m/d$ (and the estimation ratio $d/n$).
% \end{proposition}

The ratio-collapse experiment (\cref{fig:app_ratio_collapse_sweeps}) confirms that $m/d$ is the governing parameter: sweeping $m$ with $d = 10$ fixed and sweeping $d$ with $m = 3$ fixed yield overlapping curves for \rsq{}, \dciD{}, and \mccP{}.  A further experiment (\cref{fig:app_disentangling_ratios}) disentangles $m/d$ from $d/n$ by sweeping one while holding the other constant, confirming that structural dependence ($m/d$) and estimation dependence ($d/n$) are separable effects.


%% ============================================================
\subsection{Computational reliability: sample complexity and false-positive control}\label{sec:computational}
%% ============================================================

Structural misspecification is a population-level property.  In practice, metrics are computed on finite samples, introducing an additional source of error that can interact with structural regime.  We study two aspects: (i)~stability of metric scores as a function of sample size $n$, and (ii)~false-positive rates under null encoders.

\paragraph{Sample sensitivity.}  We sweep $n \in \{50, 100, 200, 500, 1000, 5000\}$ across all (DGP, encoder) combinations (\cref{fig:app_sample_sensitivity_full}).  In the easiest setting (D1$\times$E1), all metrics stabilise by $n = 200$.  Under nonlinearity (E2) or entanglement (E3), \rsq{} and \dciD{} require $n > 1000$ and exhibit high variance across seeds, while \mccS{} converges quickly and with low variance across all settings.  The interaction between structural regime and sample complexity is non-trivial: \rsq{} under D3$\times$E2 shows variance bands spanning 0.2--1.0 even at $n = 1000$ (\cref{fig:app_sample_sensitivity_full}, panel D3$\times$E2), while the same metric under D1$\times$E1 has negligible variance at $n = 50$.

\paragraph{False-positive control under null encoders.}  A metric should assign $\approx 0$ to a random encoder that carries no information about $\rvz$.  We construct two null encoders: E9 ($\hat{\rvz} \sim \mathcal{N}(\mathbf{0}, I_m)$) and E10 ($\hat{\rvz} \sim \text{Uniform}([0,1]^m)$), both independent of $\rvz$.  At large $n$, all four core metrics converge to $\approx 0$ when $m = d$ (\cref{fig:app_null_bars}).  The interesting regime is when $m$ or $m/n$ varies.

\begin{figure}[t]
    \centering
    % PLACEHOLDER: Image 19 — Phase diagram of null-encoder reliability
    \includegraphics[width=\linewidth]{figures/fig_null_phase_diagram.pdf}
    \caption{\textbf{False-positive risk depends on $m/d$ and $m/n$ jointly.}  Each cell shows the metric score under E9 (random Gaussian, independent of $\rvz$) as a function of $m/d$ (rows) and $m/n$ (columns).  Green ($\approx 0$): trustworthy.  Red ($\gg 0$): inflated.  \mccP{} is systematically inflated when $m/n > 0.1$, reaching scores $\geq 0.8$ when $m \approx n$. \rsq{} produces negative values (displayed as 0) due to overfitting of the linear probe.  \dciD{} is the most robust, remaining near zero across the grid.  See \cref{fig:app_null_convergence} for convergence curves across $n$.}
    \label{fig:null_phase}
\end{figure}

\cref{fig:null_phase} reveals that \mccP{}'s false-positive rate depends critically on the ratio $m/n$: when $m$ is comparable to $n$, the Hungarian matching over a large random correlation matrix finds spuriously high correlations.

% MATH PLACEHOLDER:
% \begin{proposition}[MCC false-positive scaling]\label{prop:mcc_fp}
% Let $\hat{\rvz} \sim \mathcal{N}(\mathbf{0}, I_m)$ independently of $\rvz \sim \mathcal{N}(\mathbf{0}, \Sigma)$, with $n$ paired samples.  The sample correlation matrix $\hat{C} \in \sR^{m \times d}$ has entries $\hat{C}_{ij} \sim \mathcal{N}(0, 1/n)$ approximately.  Then:
% \[
%   \E[\mccP] \approx \frac{1}{d} \sum_{j=1}^{d} \E\!\left[\max_{i \in \mathcal{A}_j} |\hat{C}_{ij}|\right],
% \]
% where $\mathcal{A}_j$ denotes the set of available indices after previous matchings.  For the first matched factor,
% \[
%   \E\!\left[\max_{i=1}^{m} |\hat{C}_{i1}|\right] \asymp \sqrt{\frac{2 \log m}{n}}.
% \]
% Thus $\mccP = \Omega(\sqrt{\log m / n})$, which is non-negligible when $m = \Omega(n / \log n)$.
% \end{proposition}

\paragraph{Takeaway.} Structural misspecification and finite-sample artefacts are compounding, not redundant: a metric that is structurally valid can still be unreliable at small $n$, and a metric that is structurally misspecified does not improve with more data.  Practitioners should verify both conditions.


%% ============================================================
\subsection{Synthesis: the validity map}\label{sec:phase_summary}
%% ============================================================

\cref{fig:phase} synthesises the preceding results into a single validity map, organised by latent factor structure (columns: D1, D2, D3/D4) and encoder geometry (rows: overcomplete, matched, undercomplete).  Each cell records which metrics are calibrated (blue), mixed (yellow), or systematically miscalibrated (pink) for the corresponding structural regime.

The map reveals a stark asymmetry:

\begin{quote}
\emph{Only one cell---independent factors with matched-dimension elementwise encoding (D1, $m = d$, E1/E2)---lies within the validity domain of all metrics studied.  Every departure along either axis introduces at least one systematic failure mode for at least one widely used metric.}
\end{quote}

The practical consequence is a two-step checklist for practitioners: (1)~determine the structural regime of the (latent factor structure, encoder geometry) pair using the taxonomy of \cref{sec:taxonomy}; (2)~select a metric whose validity domain covers that regime---or, if none does, report results for multiple metrics spanning different families and explicitly flag the mismatch.  We formalise this as a decision tree in \cref{sec:app_decision_tree}.