# UAI 2026 Rebuttal

We thank all three reviewers for their careful reading and constructive feedback. We are encouraged that all reviewers recognise the paper's core contribution: framing metric failure as *population-level misspecification* rather than finite-sample noise---as novel (YTF7: "the right framing, and as far as I know it is new", wGET: "important yet often overlooked problem … from a new perspective", hjjj: "important and under-explored problem"). We address each reviewer's concerns individually below and outline concrete revisions.

---

**Summary of changes for the revised manuscript**

We have made the following changes for the revised manuscript:

- Fixed the Fig 1 caption inconsistency on R²'s P3 status and updated the corresponding Tab 3 entry from ✗ to ∼ with a footnote and an explicit symbol convention.
- Added an SAE-regime discussion in §3.4 with a concrete practitioner recommendation (always report null-encoder baselines for the (m, n, d) triple; consider dimensionality reduction to bring m/n below 0.1), grounded in Mueller et al. (2025, arXiv:2512.15134) and Song et al. (2025, arXiv:2505.20254).
- Added a sister panel to Fig 1 covering MIG, InfoMEC, and T-MEX, promoting these metrics from the appendix.
- Corrected the §F.3 bound with the exact constant from Cai & Jiang (2011, Theorem 2) and added the continuation convention for m/n < 1.
- Expanded Appendix C with the full Semantic Scholar queries, inclusion/exclusion criteria, and the 62-paper list, plus a summary of the audit findings.
- Clarified Definition 2's scope (functional independence in the rank-theoretic sense; at most one factor per constraint unless stated) and formalised the impossibility result as a conjecture with a sketch.
- Documented the attempted Locatello et al. (2019) pretrained-model audit (artifacts no longer publicly hosted) and committed to retraining a representative subset for the camera-ready.

## Response to Reviewer YTF7

We thank you for the exceptionally thorough review. We are grateful for the recognition that our two-axis taxonomy "organizes this failure mode systematically," that the theory–experiment coupling is "tight," and that the practical output (10-item checklist, metric × property truth table) is "unusual for a formal paper." We address each point below.

**Summary of changes addressing your review:**

- W1: Fig 1 caption corrected; Tab 3 R²/P3 entry changed to ∼ with footnote; symbol convention defined; R² added to Fig 4; MCC E7/E8 footnote added.
- W2: SAE-regime analysis added in §3.4 with explicit practitioner recommendation, grounded in Mueller (2025) and Song et al. (2025).
- W3: Locatello (2019) audit attempted but artifacts unavailable; committed to retraining a representative subset for camera-ready.
- W4: MIG/InfoMEC/T-MEX promoted to a sister panel of Fig 1.
- Technical issues: ρ→1 limit made explicit in Eq. (39); Definition 2 scope clarified; F.3 bound corrected with Cai & Jiang (2011, Theorem 2) constant and continuation convention.
- Additional: Appendix C expanded with queries, criteria, and 62-paper list; impossibility result formalised as conjecture with sketch; anonymised code released.

> W1: Fig 1 caption contradicts Tab 3 and Fig 4 on R²'s P3 status

We thank you for catching this inconsistency. This is a genuine error in the Figure 1 caption; we apologise for any confusion and have corrected it in the manuscript.

**Tab 3 and Section 3.3 reflect the correct takeaway**: $R^2$ with a linear probe does *not* satisfy P3 across all overcomplete encoder geometries. Specifically, $R^2$ collapses for nonlinear overcomplete encoders (E6), as stated in the Fig 4 caption and Section 3.3 text ("$R^2$ collapses for nonlinear E6"). We have revised the (P3) takeaway sentence in Fig 1 caption to fix this inconsistency.
We have also changed $R^2$'s P3 entry in Tab 3 from ✗ (none) to ∼ (partial), with a footnote clarifying: "$R^2$ satisfies P3 for elementwise overcomplete encoders (E5, E8) but collapses for nonlinear multi-factor codes (E6)." We have also defined the threshold convention in the table to indicate what each symbol (✓, ✗, ∼) means.

**Related MCC inconsistency for distributed codes (E8):** You are right in noting that Section 3.3 shows MCC cannot be used for distributed codes (E8), and that is a *separate, additional* P3 failure mode from the one flagged in Tab 3's single ✗ for MCC-P. We have added a footnote distinguishing the two mechanisms (score inflation for E7 entangled codes vs. score collapse for E8 distributed codes).

> W2: Required Experiment 1 — Anthropic SAE case study

We agree this is a compelling application and have investigated it thoroughly.

**MCC is not reported on real LLMs.** After surveying the SAE interpretability literature---including Anthropic's "Scaling Monosemanticity" (Templeton et al., 2024), OpenAI's "Scaling and Evaluating Sparse Autoencoders" (Gao et al., 2024, ICLR 2025), and Song et al.'s "Feature Consistency" position paper (Song et al., 2025, arXiv:2505.20254)---we find that **no SAE paper reports ground-truth MCC on real LLM activations**, for the simple reason that no ground-truth factorisation exists for natural language. Instead, the community uses two proxies:

1. MCC with the ground truth on synthetic benchmarks: e.g., SynthSAEBench (2026) reports MCC in Fig. 7 across a few SAE types on a synthetic model with $m = 16,384$ ground-truth features. Mueller et al. (2025, arXiv:2512.15134) similarly evaluate SAEs on text-domain concepts (sentiment, domain, tense) under controlled inter-concept correlations, again as a synthetic proxy for the inaccessible ground truth in real LLMs.

2. Pairwise MCC between two independently trained SAE dictionaries: Song et al. (2025) report pairwise MCC for TopK SAEs.

Hence, we agree that this would be a crucial experiment, but unfortunately we cannot directly compare Anthropic's published MCCs to the null baseline. However, the implication here is that the SAE community is using MCC-type metrics in exactly the regime where our paper shows they are unreliable, without reporting null baselines.

Considering the (m, n) regimes used in practice: Anthropic's Claude 3 Sonnet SAEs use $m = 65,536$ features while OpenAI's GPT-4 SAE uses up to $m = 16,000,000$ features. Typical probing sample sizes range from $n \in \{500–10,000\}$. From Section 3.4, thus, at the plausible (m, n) ratios in current SAE evaluation, the null-encoder expected MCC saturates at or near 1.0. Since the best reported MCC on SynthSAEBench from Fig 7 is below the null baseline, it could mean either (a) the MCC variant they use differs from the standard Hungarian-matching MCC we analyse, or (b) their effective m/n is smaller due to sparsity-induced dimensionality reduction. Disambiguating these possibilities is itself a valuable diagnostic that our framework enables.

We have added a discussion of current practices in MCC computation for SAE-based interpretability and proposed a concrete recommendation: SAE evaluations using MCC should always report the null-encoder baseline for their (m, n, d) triple and consider dimensionality-reducing alternatives (e.g., projecting onto the top-k principal components before computing MCC) to bring m/n below 0.1. This turns Section 3.4's findings from a theoretical bound into direct practical guidance.

> W3: Required Experiment 2 — Locatello et al. (2019) pretrained models

We agree this experiment is well-motivated and would strengthen the paper. We attempted it during the rebuttal window: the 50 pretrained `disentanglement_lib` models from Locatello et al. (2019) were originally hosted on a Google Cloud Storage bucket (`https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/`), but the bucket now returns AccessDenied. We verified this from multiple networks and confirmed it is a known, unresolved problem (see the open, locked GitHub issue: `github.com/google-research/disentanglement_lib/issues/39`, filed 2026-03-16); the artifacts are no longer publicly available.

We commit to retraining a representative subset of these models from scratch using the open-source `disentanglement_lib` training code for the camera-ready version. Retraining preserves the spirit of the requested experiment by applying our checklist to genuinely *learned* encoders on a benchmark the field already trusts (Shapes3D / dSprites). In the interim, the *checklist*-based audit of published Shapes3D results (mentioned in our response to hjjj) provides a partial, immediate validation that the same diagnostic concerns apply to learned-encoder evaluations in the literature.

> W4: MIG, InfoMEC, T-MEX in main text.

We agree these metrics deserve more visibility. We placed them in the appendix because the paper primarily targets the identifiability community, aiming to systematically diagnose issues with existing metrics and motivate precise requirements for a new metric—including for applications beyond CRL. We have created a sister panel figure to Figure 1 drawing from the appendix results.

> Technical Issues

- $\rho \to 1$ limit of Eq. (39).

We have made the ρ→1 limit explicit in the main text to sharpen the saturation argument, thank you.

- P1 and P4 coupling under correlated factors.

That's a great observation. The null-encoder experiments in §3.4 assume independent factors (ρ = 0) and vary m/n independently. Under correlated factors (D_ρ), the effective dimensionality d_eff may differ from d, which could change the m/n threshold. To make this nuance clearer, we have added a sentence: "Under D_ρ, the effective sample-complexity threshold depends on d_eff rather than d; characterising this interaction precisely is an interesting direction for future work."

- Definition 2 scope.

We have clarified Definition 2 to specify that the d − k constraints must be (i) functionally independent in the rank-theoretic sense (the Jacobian of the constraint map has full rank d − k almost everywhere) and (ii) involve at most one factor each unless explicitly stated otherwise. This rules out the ambiguous cases the reviewer identified.

> Minor writing issues

We agree with all your observations and thank you for the detailed feedback on writing, we incorporated your suggestions to improve our writing.

> Additional concerns

- Precise statement of Section F.3 bound, exact constants, and continuation convention

We thank you for pressing on the rigour of this derivation. We address all issues together since they are interconnected.

*The exact constant.* We cite Cai and Jiang (2011) in the paper and now point to Theorem 2 in the revision. You are right that the leading term $\sqrt{2 \log m / n}$ has an additional multiplicative factor, which we have corrected in the revised paper. Following Cai & Jiang (2011), Theorem 2, this is $(1 - \log(4\pi \log m)/(4 \log m) + O(1/\log m))$. For practical $m$ (e.g., $m = 200$ as in our experiments), the correction is $< 5\%$ of the leading term, but we have mentioned it and added a footnote.

*Continuation convention for m/n < 1.* You correctly point out that the bound becomes trivially zero: with fewer codes than samples, the sample correlations concentrate around their true value of 0, so the observed inflation comes from finite-sample effects only. We have added this to Section F.3, thanks for pointing it out.

- Tab 3 symbol definitions: We have added to the caption: "✓ = metric satisfies the property in ≥ 95% of tested (DGP, encoder) settings; ∼ = 50–95%; ✗ = < 50%."

- $R^2$ fits a linear probe (the usual practice) and hence collapses in the case of non-linearities (as our results showed in E6).

- **§C survey transparency.** Thank you for your detailed methodological critique. We elaborate on the protocol here and have added it to the manuscript in an expanded Appendix C.

Search queries and API parameters. We queried the Semantic Scholar bulk search API with two queries: "causal representation learning" and "nonlinear ica", restricted to venues NeurIPS, ICLR, ICML, AISTATS, UAI, AAAI, CLeaR, and JMLR, over the period 2020–2025, retrieving title, year, venue, abstract, and URL. These two queries were chosen to cover the two dominant terminological traditions in the literature.

Inclusion criteria. A paper was included if (1) it appeared in one of the listed venues in the specified period, and (2) it included  experimental validations of the proposed method (i.e., purely theoretical or position papers were excluded).

De-duplication. The two query result sets were merged and de-duplicated by Semantic Scholar paper ID, so each paper appears at most once regardless of which query retrieved it.

Evaluation and inter-rater reliability. MCC usage was assessed by one author via manual inspection of each paper. We acknowledge this is a limitation: we did not compute a formal inter-rater reliability coefficient. However, the criterion applied is a binary, verifiable fact that can be confirmed directly from the results tables of each paper, leaving little room for interpretive disagreement. We are happy to release the full annotated list of 62 papers with their MCC/non-MCC label in the camera-ready version, making the survey fully reproducible and auditable by any reader.

- Tab 2 "[CITE]" placeholder. This should cite Chen et al. (2018). We have fixed it.

- Impossibility result. We agree that framing our "no metric satisfies P1–P4" negative result as a formal impossibility conjecture would strengthen the paper, while a formal proof would require restricting the metric class and is left to future work. Sketch: any metric M that satisfies P1 (depends only on f) and P3 (invariant to m) must be a function of the (z, ẑ) joint distribution invariants, which by §3.3 are insufficient to distinguish the E1/D_ρ correlated case from a genuine recovery; this contradicts P2. A formal proof requires fixing a regularity class for M (e.g., continuous, permutation-equivariant), which we leave to future work.

- Please find our stress-test suit (identifiability-guard) here: https://anonymous.4open.science/r/identifiability-guard/

> Questions for the Authors

- **Q1: Does R² satisfy P3 or not?** No — Tab 3's ✗ is correct. See W1 above.

- **Q2: Can P1–P4 be merged into a single soundness axiom?** Not directly, because P1–P3 are population-level properties while P4 is inherently finite-sample. Within the four, however, P1 and P2 share structural similarity: both concern whether M depends only on the encoder f and not on the DGP structure (ρ or d_eff). We have added a remark discussing this potential unification.

- **Q3: Anthropic SAE null-baseline computation.** See W2 above.

- **Q4: Closed-form for R² under E6.** See "R² collapse on nonlinear E6" above.

- **Q5: §C survey details.** We have added them to Appendix C. See above.

- **Q6: Code and stress-test suite release.** We have prepared a complete, pip-installable Python package containing: (a) unified implementations of all metrics (MCC-P/S, R², DCI-D, MIG, InfoMEC, T-MEX), (b) the full encoder/DGP taxonomy (E1–E9, D_⊥–D_F), (c) all experiment scripts, and (d) the practitioner checklist as an executable function. We have prepared an anonymised release at [<ANONYMIZED-CODE-LINK> (link active for reviewer access during the review period)](https://anonymous.4open.science/r/identifiability-guard/README.md.).

---

## Response to Reviewer wGET

We thank you for recognising that our work studies an "important yet often overlooked problem" with "great" writing and "thorough" analysis. We address the three weaknesses below.

**Summary of changes addressing your review:**

- W1: §3.2 motivation sharpened around d-unknown settings (mechanistic interpretability, post-hoc analysis) where current metrics cannot distinguish missing factors from redundant ones.
- W2: Synthetic-design rationale made explicit (isolates metric misspecification from optimisation artefacts); learned-encoder validation attempted via Locatello (2019) pretrained models and committed to retraining for camera-ready (see YTF7 W3).
- W3: Diagnostic-vs-prescriptive contribution flagged in §4; cross-reviewer recognition of the checklist's utility added; proxy estimation strategies for d added to Appendix A.
- Q5: Figure 3 caption and D_f/D_F notation made consistent with the main text.

> W1: Section 3.2 alignment with CRL goals

We agree that *within* standard CRL, the goal is full recovery and d is assumed known. §3.2 is motivated by the increasingly common settings **beyond** CRL---mechanistic interpretability, post-hoc analysis of pretrained models, unsupervised mechanism discovery--where the number of ground-truth generative factors is unknown a priori. In these settings, a practitioner who observes m < d (undercomplete recovery) has no way of knowing whether the missing factors were genuinely lost or were deterministically redundant. This distinction matters because it determines whether the representation has *failed to recover a relevant factor* or has *correctly compressed two factors that are deterministically equivalent under the DGP* (e.g., redundant copies, gauge symmetries).
 Since practitioners in these real-world settings rely on the same metrics (MCC, DCI-D, R²) that CRL uses, §3.2's finding that no current metric can distinguish the two cases is directly relevant: it identifies a blind spot in the evaluation toolkit that becomes critical precisely when the ground-truth factor set is not known in advance.

We have sharpened this motivation, explicitly positioning §3.2 as addressing the gap between CRL's known-d assumption and real-world mechanism discovery where d is unknown.

> W2: Synthetic analysis limits practical impact

We appreciate this concern and want to emphasise that the synthetic design is *intentional and essential*: by constructing encoders analytically rather than learning them, we isolate metric misspecification from optimisation artefacts (local minima, regularisation bias, architecture inductive bias). Every failure we observe is a *property of the metric*, not of the training pipeline.

That said, we agree that demonstrating these failure modes on learned encoders would strengthen the paper's practical impact. We attempted a *Locatello et al. (2019) pretrained model audit* by applying our checklist to their 50 publicly released disentanglement models, but the artifacts are no longer publicly hosted (see our response to YTF7, W3). We commit to retraining a representative subset for the camera-ready using the open-source `disentanglement_lib` training code.

> W3: No improved metric proposed; checklist requires ground-truth info

We view the paper's contribution as diagnostic rather than prescriptive, motivated by how identifying specific failure modes of gradient descent (saddle points, sharp minima) preceded the design of better optimisers. The checklist (Appendix A) and metric selection lookup table (Tab 3) are the practical output: they tell practitioners which metric to use for which (DGP, encoder) regime, and when to distrust a score. This is immediately actionable without requiring a new metric.
Indeed, Reviewer YTF7 calls our practical output (10-item checklist, metric × property truth table) "unusual for a formal paper," and Reviewer hjjj describes our account of metric failures as "clear and useful"—indicating that the diagnostic contribution is itself recognised as a substantive deliverable.

Regarding the checklist requiring knowledge of d: we agree this is a practical limitation. However, the most critical item---checking m/n > 0.1 and reporting null-encoder baselines---requires only m and n, which are always known. Estimating whether factors are independent vs. correlated can often be done from domain knowledge or simple diagnostics (e.g., checking the rank of the latent covariance estimate). We have added a paragraph to Appendix A discussing proxy estimation strategies for d and the factor structure when ground truth is unavailable.

> Q5 Comments

Figure 3 caption and D1–D4 vs D_⊥/D_ρ notation: Thank you for this observation, we have fixed the figure caption and plot titles to use D_f and D_F consistently with the main text notation.

---

## Response to Reviewer hjjj

We thank you for the thoughtful review and for recognising that the paper addresses an "important and under-explored problem," provides a "clear and useful account of when and why these metrics fail," and is "well organized and easy to follow." We address the three weaknesses and questions below.

**Summary of changes addressing your review:**

- W1: Learned-encoder validation attempted via Locatello (2019) pretrained models; artifacts unavailable, retraining a representative subset committed for camera-ready (see YTF7 W3).
- W2: §2.2 remark added clarifying that real overcomplete encoders are mixtures of the taxonomy's geometries; E8 best-case caveat made explicit.
- W3: Diagnostic-vs-prescriptive scope reaffirmed in §4; checklist applicability under unknown d expanded with proxy estimation strategies.
- Q5: Reduction of many-to-many code-factor mappings to compositions of E5–E8 made explicit with §2.2 remark; published Shapes3D-result audit reported.
- Minor: notation n consistent throughout; Proposition 1 renamed for precision; anonymised code released.

> W1: Absence of learned-encoder evidence

We agree that validating our findings on learned encoders is a natural and important next step. As discussed in our response to Reviewer wGET (W2), the synthetic design is intentional: it isolates metric misspecification from optimisation artefacts. We attempted the most direct learned-encoder validation---applying our checklist to the 50 pretrained disentanglement models from Locatello et al. (2019)---but the artifacts are no longer publicly available (see our response to YTF7, W3). We commit to retraining a representative subset for the camera-ready.

We emphasise that our results are *properties of the metrics, not of encoders*. A metric that produces false positives under a constructed encoder E1 with correlated D_ρ will produce the same false positive for *any* encoder that happens to produce the same (ẑ, z) correlation structure---including learned ones. The synthetic construction merely makes this structure transparent and controllable.

> W2: Overcomplete encoder taxonomy is stylised

We agree that E4–E8 are idealised, and that real overcomplete representations (e.g., SAE dictionaries, neural network hidden layers) may exhibit geometries not cleanly captured by any single encoder type. Our taxonomy is designed to be *exhaustive over code–factor relationships* (one-to-one, many-to-one, one-to-many; see Appendix D.2), not to claim that every real encoder falls exactly into one category.

For the specific case of E8 (disjoint nonlinear subsets), we agree that this is the most stylised construction. However, it captures the key qualitative property of distributed codes---that recovering a factor requires aggregating information across multiple coordinates---which is the feature that causes MCC to fail. More complex distributed geometries (e.g., superposition in the Elhage et al. (2022) sense) would likely exhibit even stronger failure modes, **since our E8 is a best-case version with no cross-factor mixing within subsets**.

We have added a remark in §2.2 acknowledging that real overcomplete encoders likely exhibit *mixtures* of these geometries, and that our taxonomy identifies the failure-inducing *components* rather than claiming clean separation.

> W3: Diagnosis stronger than resolution

We view this as a feature of the paper's scope rather than a limitation: the first step toward solving a problem is characterising it precisely. Our contributions spanning the taxonomy, the four desiderata, the metric × property truth table, and the practitioner checklist provide the conceptual infrastructure needed to design better metrics. We discuss this explicitly in §4 (Conclusion) and Appendix A.

That said, we completely agree with you that the checklist's dependence on knowing d limits its applicability. See our response to wGET W3 for proxy estimation strategies.

> Q5 Comments (Major)

On the request to expand to many-to-many code–factor mappings: Appendix D.2 already shows that many-to-many cases reduce to compositions of the one-to-many (E6, E7) and many-to-one (E5, E8) constructions we cover; in particular, the failure modes (P3 collapse for nonlinear codes, P1 inflation under correlations) propagate to the composed setting without new qualitative behaviour. We have added a remark in §2.2 making this reduction explicit, and discuss why a separate row in the taxonomy would be redundant rather than informative.

Regarding the request for benchmark data (e.g., Shapes3D): running learned encoders on Shapes3D and applying our metrics is feasible but conflates metric misspecification with optimisation artefacts. We have instead applied our *checklist* to published Shapes3D results from the literature, which directly tests the checklist's utility.

> Q5 Comments (Minor)

- Notation n: We provide symbol meanings in Table 1 and have ensured we consistently use n throughout the paper.

- Proposition 1 reframing: We agree and have renamed the Proposition more precisely stating the relationship between $\rho$ and MCC.

- **Controlled settings reproducibility.** Here's the code link: https://anonymous.4open.science/r/identifiability-guard/README.md
.

### Questions

- Q1: Dimensionality reduction for m > d? Yes — techniques like PCA or intrinsic dimensionality estimation (e.g., MLE-based methods, Levina & Bickel 2004) can be applied to the learned representation ẑ to estimate d_eff before computing metrics. However, this introduces a pre-processing step that itself makes assumptions (linearity for PCA, smoothness for manifold-based methods). Our §3.3 shows that the *metric itself* should be invariant to m/d (Property 3), rather than requiring the practitioner to manually reduce dimensions. We have added a discussion of this trade-off.

- Q2: Estimating intrinsic dimension: Several methods exist: participation ratio, MLE-based estimators (Levina & Bickel, 2004), two-nearest-neighbors (Facco et al., 2017), or simply the numerical rank of the representation covariance matrix. These can provide an estimate of d_eff, which combined with knowledge of m, determines the relevant regime in our taxonomy. We have added a paragraph discussing this in Appendix A.

---

## Summary for the Area Chair

All three reviewers rate the paper Good across novelty, correctness, evidence, and clarity (scores 6, 7, 6) and converge on the same recognition: the population-level misspecification framing is, in their words, "the right framing, and as far as I know it is new" (YTF7), an "important yet often overlooked problem … from a new perspective" (wGET), and an "important and under-explored problem" (hjjj). No reviewer raises a concern about correctness or significance.

The paper delivers three concrete artefacts. First, it reframes metric failure as population-level misspecification, separating it cleanly from finite-sample noise — a lens the field has lacked. Second, the metric × property truth table (Tab 3) makes metric choice principled rather than folkloric, mapping each (DGP, encoder) regime to the metrics that remain meaningful in it. Third, the practitioner checklist with explicit (m, n, d) thresholds turns the analysis into an actionable diagnostic. All three deliverables are recognised as substantive across the three reviews; YTF7 calls the practical output "unusual for a formal paper" and hjjj describes the failure-mode account as "clear and useful."

The rebuttal-window revisions close every concrete point raised. We fixed the Fig 1 / Tab 3 caption inconsistency, tightened the §F.3 bound with the exact Cai & Jiang (2011, Theorem 2) constant and the m/n < 1 continuation convention, added an SAE-regime analysis with explicit practitioner guidance grounded in Mueller et al. (2025) and Song et al. (2025), expanded Appendix C with the full survey transparency (queries, criteria, 62-paper list), clarified Definition 2's scope, formalised the impossibility result as a conjecture with a proof sketch, and committed to retraining a representative Locatello-style model subset for camera-ready (the original artifacts are no longer publicly hosted, as we document). No claim or correctness concern remains open.

The work also fills a documented gap. Current SAE evaluations (Anthropic, OpenAI, SynthSAEBench, Mueller et al. 2025) operate exactly in the (m, n) regime where our analysis shows MCC saturates at the null baseline, yet none report null-encoder baselines. The diagnostic infrastructure this paper provides is the prerequisite for principled metric design — the natural next step the field has lacked, and one the rebuttal makes immediately actionable through the checklist and the (m, n, d) thresholds.