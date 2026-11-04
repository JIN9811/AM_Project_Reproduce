# Harandi manuscript experiment inventory

Source: Harandi et al., DOI **10.1002/nme.7388**, printed pages 1–37. Audit covers Sections 4.1–4.4, Figs. 1–25, Tables 1–4, and Appendices A/B. Page numbers below are printed manuscript pages. The companion JSON is the detailed machine-readable inventory.

**This is an inventory, not a claim that the experiments were executed.** Printed values, contextual defaults, graphical inferences, and missing values are kept distinct. No existing project code was changed and no GPU training was run.

## Executable families and coverage

| Study | Required case or comparison | Source |
|---|---|---|
| S01 | Geometry1 mixed PINN, coupled | p.11,12,13,15,17,19 Table1; Sections4.1–4.1.2; Fig.11 |
| S02 | Geometry1 mixed PINN, sequential | p.11,12,13,15,17,19 Table1; Sections4.1–4.1.2; Fig.11 |
| S03 | Geometry1 standard PINN, coupled | p.11,12,13,15,17,19 Table1; Sections4.1–4.1.2; Fig.11 |
| S04 | Geometry1 standard PINN, sequential | p.11,12,13,15,17,19 Table1; Sections4.1–4.1.2; Fig.11 |
| S05 | Geometry1 DEM, coupled | p.11,12,13,15,17,19 Table1; Sections4.1–4.1.2; Fig.11 |
| S06 | Geometry1 DEM, sequential | p.11,12,13,15,17,19 Table1; Sections4.1–4.1.2; Fig.11 |
| S07 | Energy-term omission observation / no-energy counterexample | p.13 Section4.1.1, paragraph following coupled-loss discussion |
| S08 | Geometry2 mixed PINN, coupled soft boundaries | p.11,18,19,20,21,22,23 Table1; Section4.2.1; Figs.12–14,16 |
| S09 | Geometry2 mixed PINN, sequential soft boundaries | p.11,20,22,23 Table1; Section4.2.2; Fig.16 |
| S10 | Geometry2 left-boundary collocation refinement | p.20,23 Section4.2.1; Fig.15 |
| S11 | Geometry2 hard Dirichlet boundaries with Adam | p.24,25,26 Figs.17–18; Eqs.62–64; Section4.3 |
| S12 | Geometry2 hard Dirichlet boundaries with L-BFGS | p.24,25,26,35,36,37 Figs.17–18; Section4.3; AppendixB Figs.B1–B2 |
| S13 | Geometry2 depth accuracy sweep | p.25,26,27 Section4.3; Figs.19–20; Table4 |
| S14 | Geometry2 width accuracy sweep | p.25,26,27 Section4.3; Figs.19,21 |
| S15 | Architecture runtime benchmark (25 layer/width pairs) | p.26,27 Section4.3; Table4 |
| S16 | One-million Adam versus ten-thousand hard L-BFGS runtime claim | p.26 Remark7 |
| S17 | Transfer/parametric learning: physics-data weight comparison and extrapolation | p.27,28,29,30,31 Section4.4; Eq.65; Figs.22–25 |
| S18 | Finite-element mesh convergence | p.34,35 AppendixA; Fig.A1 |
| S19 | AppendixB complete hard-constraint full fields | p.35,36,37 AppendixB; Figs.B1–B2 |

There is no defensible single exact count of independent full training runs: Appendix B may reuse the hard L-BFGS configuration, the default 5×40 architecture appears in both sweeps, the timing table does not say all 25 cells are full 10k runs, and the no-energy ablation/fourth physics weight are incompletely specified.

## Baseline settings

- **Physics/domain:** stationary plane-strain thermoelasticity on [0,1]²; prose/axes use mm, but Fig. 2 labels dimensions μm. Temperature-independent elastic coefficients. Numerical α and T₀ are not supplied.
- **Geometry 1:** one circle, diameter 0.3; center not dimensioned. Approximately 5,000 nonuniform/enriched training points. Evaluation grid count not stated. Matrix/inclusion E = 0.3/1.0 GPa, k = 0.3/1.0 W/(m K), both ν = 0.3 (Table 2, p.13).
- **Geometry 2:** four rectangles: upper left 0.28×0.20, upper right 0.52×0.42, lower left 0.24×0.32, lower right 0.42×0.24; origins not dimensioned. Train 2,601, evaluate 10,201; 51²/101² grids are figure/count-supported inference. Matrix/inclusion E = 1.0/0.5 GPa, k = 1.0/0.5 W/(m K), both ν = 0.3 (Table 3, p.20).
- **BCs:** ux=0 on left/right; uy=0 on bottom/top; T=1 left and T=0 right; qy=0 top/bottom. Expanded mechanical Neumann loss is zero (Eq.56).
- **Mixed network:** eight separate scalar networks for ux, uy, σxx, σyy, σxy, T, qx, qy; 5 hidden layers × 40 neurons each, tanh, full batch. Standard PINN/DEM predict primary u,T and derive stress/flux; both use 5×40 tanh but exact scalar branch count is not independently stated.
- **Adam baseline:** learning rate 10⁻³; coupled 100,000 epochs; sequential 200,000 total = 100,000 per field, alternating thermal then mechanical blocks of 20,000, five cycles, inactive field frozen. Table 1 p.11. Geometry 2 later has an unresolved 1,000,000-Adam-iteration timing claim in Remark 7.
- **Loss:** mixed strong + energy + constitutive connection + boundary terms. Global energy uses MAE; point losses use MSE. There is no specified energy-specific multiplier or energy-weight sweep. Figure 6 energy plateau does not authorize discarding energy.

## Full study details

Geometry 1 requires all **six** mixed/standard PINN/DEM × coupled/sequential cases. Figure 11 compares ux, σxx, T, qx at x=0.5 mm. Standard PINNs miss heterogeneous primary-field responses, and DEM remains inferior to FE. The energy-omission observation (p.13) is a separate qualitative ablation: no figure, numeric errors, or exact omitted-term configuration is supplied.

Geometry 2 includes mixed coupled and sequential baselines, then approximately 1,000 additional left-boundary points (Fig.15). Maximum ux and T error reduction factors are 1.6 and 3.0. The refinement is not the hard-boundary study, and its separate training budget/initialization is not stated.

Hard boundaries use ûx=ux·x(x−1), ûy=uy·y(y−1), T̂=T·x(x−1)+1−x (Eqs.62–64). Figures 17–18 compare **hard** Adam and **hard** L-BFGS for 10,000 iterations using 2,601 collocation points; prior soft baselines provide context. L-BFGS line search, history and tolerances are not specified. Appendix B supplies all eight full fields for coupled hard L-BFGS after 10,000 iterations.

The accuracy depth sweep is **[1,2,4,5,6]** hidden layers at 40 neurons, not every integer depth 1–6. The width sweep is **[5,10,20,40,60]** at five layers. The context is hard coupled L-BFGS, tanh, 10,000 iterations. Figure 19 explicitly displays **2×40, 5×20, 5×40, 6×40**. Figure 20 caption is wrong; use its depth x-axis and body text. Five layers has a localized **σxx maximum-error exception**; larger width is not uniformly better.

Table 4 benchmarks all 25 combinations of depths [1,2,4,5,6] and widths [5,10,20,40,60], averaging 100 L-BFGS iterations on Apple M2 Pro. Accuracy curves cover only two slices through that grid. Times are recorded below.

| Hidden layers | 5 neurons | 10 | 20 | 40 | 60 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.24670 | 0.24920 | 0.25260 | 0.27120 | 0.28210 |
| 2 | 0.25950 | 0.26685 | 0.27060 | 0.27910 | 0.28845 |
| 4 | 0.29360 | 0.29480 | 0.30550 | 0.31865 | 0.33300 |
| 5 | 0.31410 | 0.31360 | 0.32940 | 0.34090 | 0.36000 |
| 6 | 0.32840 | 0.33460 | 0.34930 | 0.37180 | 0.39740 |

Values are seconds per epoch/iteration. The 5×40 entry implies approximately 56.8 minutes for 10,000 identical iterations, whereas Remark 7 says about 40 minutes. These are manuscript claims, not current-machine speed targets.

**Parametric/transfer study:** Fig.22 explicitly defines Eᵣ=E_matrix/E_inclusion and kᵣ=k_matrix/k_inclusion, jointly varied through paired values (1,1),…,(10,10), with an unseen (15,15) test. This is not a 10×10 independent ratio grid. Train 10,000 epochs at each ratio, initialize the next from the previous, giving an inferred 100,000 epochs per weight if all ten stages are used. The prose does not independently specify order, optimizer, data count, data-loss normalization, absolute material scale, or hard/soft BC handling.

Equation 65 is L_total=L_data+w(L_T+L_M): **w weights all physics, not energy alone**. Four weights over [0,1] are claimed, but only 0 and 0.1 are identified for plotted predictions; 1 is the stated upper endpoint and the remaining exact value is absent. Do not invent an exact four-value source set. Figures 23/24 show w=0.1/0 at ratio15. Figure25 panels and p.31 say x=0.35 mm, although its caption says0.5.

The physics+data exception must remain visible: qx maximum error is slightly worse, with plotted upper error limits **0.2787 versus 0.2573** W/(m K), despite improved primary fields and interface behavior. Pure data-driven transfer is reported about **82% cheaper** (Remark8), with no absolute times.

**Appendix A:** Geometry2 FE meshes have 50,100,200 elements per edge; compare ux,σxx,T,qx sections at x=0.5. Authors call 100 nearly converged, but no quantitative tolerance is supplied and the temperature curves still differ visibly.

## Reported baseline average relative errors

| Geometry / method | ux | σxx | T | Flux as written |
|---|---:|---:|---:|---:|
| Geometry1 mixed coupled | 0.59% | 0.87% | ≈0.30% | qx≈0.76% |
| Geometry1 mixed sequential | 0.81% | 0.80% | 0.26% | qy=0.66% |
| Geometry2 mixed coupled | 1.60% | 3.30% | 1.30% | 1.80%, contextual flux |
| Geometry2 mixed sequential | 0.49% | 2.12% | 0.70% | qy=1.82% |

The relative-error normalization is not defined sufficiently to reproduce these percentages exactly. Preserve qy labels from prose even though section plots show qx. Sequential Geometry1 ux is worse than coupled, so universal improvement is unsupported.

## Every figure and table accounted for

| Figure | Page | Content | Study records | Specific note |
|---|---:|---|---|---|
| 1 | 4 | Conceptual sequential/coupled training and mixed output architecture | setup | Concept diagram, not another numerical experiment. |
| 2 | 6 | Geometry1 circle and Geometry2 four rectangles | setup | Dimension labels are incomplete for positions; unit conflict recorded. |
| 3 | 6 | Mechanical and thermal boundary conditions | setup | Generic illustrative microstructure, not a third tested geometry. |
| 4 | 8 | Eight separate networks, losses and sequential/coupled flowcharts | setup | Default5×40; Fourier sign discrepancy recorded. |
| 5 | 13 | Geometry1 training, evaluation and FE meshes | S01, S02, S03, S04, S05, S06 | Approx5000 training points; evaluation count unspecified. |
| 6 | 14 | Geometry1 coupled thermal/mechanical loss histories | S01 | 100k epochs; energy plateaus; nonzero plotted mechanical NBC conflicts with Eq56. |
| 7 | 15 | Geometry1 undeformed/deformed configurations | S01 | PINN/FE deformation display scaling not stated. |
| 8 | 16 | Geometry1 coupled all five mechanical fields +FE+difference | S01 | ux,uy,sigma_xx,sigma_yy,sigma_xy. |
| 9 | 17 | Geometry1 coupled all three thermal fields +FE+difference | S01 | T,qx,qy. |
| 10 | 18 | Geometry1 sequential active-field loss histories | S02 | Each plotted history100k active epochs;200k total;20k switching. |
| 11 | 19 | Geometry1 methods × training modes cross-section | S01, S02, S03, S04, S05, S06 | x=.5, fields ux,sigma_xx,T,qx; six model curves plus FEM. |
| 12 | 20 | Geometry2 training,evaluation,FE discretizations | S08, S09 | 2601train,10201eval; upper-left FE rectangle position visually differs. |
| 13 | 21 | Geometry2 coupled all mechanical fields +FE+difference | S08 | Soft boundary error at left. |
| 14 | 22 | Geometry2 coupled all thermal fields +FE+difference | S08 | Do not mistake p.15's Figure14 citation for Geometry1 sequential full fields. |
| 15 | 23 | Geometry2 left-refined collocation and ux/T error maps | S10 | About1000 added points; maximum error reduction factors1.6 and3.0. |
| 16 | 23 | Geometry2 coupled/sequential cross-section | S08, S09 | x=.5, ux,sigma_xx,T,qx. |
| 17 | 24 | Hard constraints with L-BFGS versus Adam field/error maps | S11, S12 | ux,uy,T; both hard. Earlier soft results provide contextual comparison. |
| 18 | 24 | Hard-constraint Adam versus L-BFGS loss histories | S11, S12 | Both10000iterations. |
| 19 | 25 | Selected architecture full-field error maps | S13, S14 | Columns2×40,5×20,5×40,6×40; rows ux,sigma_xx,T,qx;10000iterations,tanh. |
| 20 | 26 | Depth sweep maximum/average errors | S13 | Depths1,2,4,5,6 fixed40neurons; caption is incorrect. |
| 21 | 27 | Width sweep maximum/average errors | S14 | Widths5,10,20,40,60 fixed5layers. |
| 22 | 27 | Parametric/transfer scheme and ratio training set | S17 | E_r=k_r in{1,...,10}; matrix/inclusion ratio explicitly defined. |
| 23 | 28 | Unseen ratio15 physics+data full fields and error | S17 | w=.1; ux,sigma_xx,T,qx. |
| 24 | 29 | Unseen ratio15 data-only full fields and error | S17 | w=0; ux,sigma_xx,T,qx. |
| 25 | 30 | Unseen ratio15 physics+data versus data-only section | S17 | Panel titles/prose x=.35; caption x=.5. |
| A1 | 35 | Geometry2 FE mesh-convergence sections | S18 | 50,100,200elements per edge; x=.5. |
| B1 | 36 | Hard-BC L-BFGS full mechanical fields | S12, S19 | 10k; five mechanical outputs. |
| B2 | 37 | Hard-BC L-BFGS full thermal fields | S12, S19 | 10k; three thermal outputs. |

| Table | Page | Content |
|---|---:|---|
| 1 | 11 | Default architecture,Adam learning rate/full batch,budgets and sequential switching |
| 2 | 13 | Geometry1 E,nu,k of both phases |
| 3 | 20 | Geometry2 E,nu,k of both phases |
| 4 | 27 | 25 L-BFGS architecture timing cells |

## Source contradictions and unresolved details

**C01 — Geometry units (pp. 6, 12, 19, 23).** Fig.2 says dimensions in μm; domain text and axes say mm. Use explicit implementation choice and disclose scale; do not call unit choice unambiguous.

**C02 — Missing thermal constants (pp. 4, 11, 13, 20).** alpha and T0 are defined symbolically but no numerical values given in material tables or results text. Author code or a declared assumption is required for exact mechanical reproduction.

**C03 — Incomplete geometry coordinates (pp. 6, 13, 20).** Circle center and rectangle origins are not dimensioned. Keep image-digitized/source-code values distinct from printed dimensions.

**C04 — FE/PINN geometry illustration mismatch (pp. 20).** Fig.12 upper-left inclusion reaches the left edge in FE panel but is interior in PINN train/test panels. Do not assert every graphical reference uses an identical material map without author-source verification.

**C05 — Adam epoch budget mismatch (pp. 11, 26).** Table1 lists100000coupledepochs; Remark7 says one million Adam iterations on Geometry2. Retain both claims; cannot infer whether extra timing run or typographical error.

**C06 — Wall-time arithmetic mismatch (pp. 26, 27).** Table4 5×40=.3409s/iteration gives3409s≈56.8min for10k, while Remark7 says≈40min. Benchmark local hardware independently; do not use manuscript timing as expected wall-clock guarantee.

**C07 — Fig.20 caption wrong (pp. 26).** Plot x-axis and prose vary hidden layers1,2,4,5,6 at40neurons; caption says varying neurons with5layers40neurons. Use plot/prose depth sweep; omit untested depth3.

**C08 — Fig.25 section coordinate mismatch (pp. 30, 31).** Caption saysx=.5mm; all panel titles and body text sayx=.35mm. Usex=.35 for displayed reproduction and record caption conflict.

**C09 — Contradictory training ranking (pp. 17).** One sentence says coupled slightly better; next says sequential better regardless of method. Report field-specific metrics; no universal sequential-superiority claim.

**C10 — Flux direction / figure citation mismatches (pp. 15, 22).** Geometry1 sequential prose references Fig.14 (actually Geometry2 coupled) and gives qy error, while comparison figures use qx; Geometry2 sequential similarly reports qy but plots qx. Preserve printed direction; do not relabel reportedqypercentages asqx.

**C11 — Parametric input and ratio ambiguity (pp. 27, 28, 29).** Fig.22 defines matrix/inclusion ratios and ratio inputs; text speaks informally of inclusion/matrix and local E,k inputs. Use explicit Fig.22 ratio convention; document input encoding choice separately.

**C12 — Incomplete physics-weight sweep (pp. 30).** Four weight cases in[0,1] claimed; onlyw=0 andw=.1 identified for plotted predictions; upper endpoint1mentioned, fourth value absent. Never invent a complete four-value source set; a chosen fourth value is an extension.

**C13 — Eq.20 qy network typo (pp. 7).** Both qx and qy are written with N_qx, although separate qy branch shown in Fig.4. Use separate qy network as architecture shows, record typographical repair.

**C14 — Fourier sign mismatch (pp. 8, 5, 11).** Fig.4 writesq=k gradT; Eq.12 and Eqs.50–51 use negative sign. Follow explicit Fourier equations, document diagram typo.

**C15 — Constitutive strain inconsistency (pp. 4, 11).** Eq.48 sigma_y uses total eps_x,eps_y while Eq.47 uses elastic strains and Eq.3 uses eps-eps_thermal. Implement tensor constitutive relation consistently or preserve exact-formula ablation separately.

**C16 — Shear convention inconsistency (pp. 4, 11).** Eq.40 defines eps_xy=.5*(uy_x+ux_y); Eq.49 multiplies that byE/[2(1+nu)], which is the engineering-shear modulus coefficient and differs from tensorial sigma_xy=2mu eps_xy. State shear convention and repair explicitly; do not silently advertise literal-equation fidelity.

**C17 — Mechanical Neumann loss inconsistency (pp. 12, 14, 18).** Eq.56 says L_M^NBC=0; Figs.6 and10 show a nonzero mechanical NBC curve. Actual comparator/training boundary losses require author-source verification.

**C18 — Equal quadrature/nonuniform points (pp. 12, 13).** p.12 justifies equal quadrature weights via uniform points; Geometry1 uses explicit enrichment and nonuniform Fig.5 points. Separate literal equal-point-weight loss from geometrically weighted energy integration.

**C19 — Inactive thermal-loss discussion vs frozen independence (pp. 9, 15).** Theory freezes inactive networks and stationary thermal losses depend only on thermal outputs; p.15 says optimizing one field may increase inactive losses, specifically thermal. Do not manufacture coupling; report actual frozen-field behavior from implementation.

**C20 — Outlook stale L-BFGS claim (pp. 31).** Conclusion proposes comparing L-BFGS in future despite Section4.3 and AppendixB already doing so. Treat as editorial inconsistency, not evidence to omit existing L-BFGS studies.

**C21 — Hard-run identity unclear (pp. 24, 36, 37).** Fig.17 error scales/results differ from AppendixB; architecture/seed identities are not separately specified. Do not count them as provably independent runs or demand identical numerical errors.

**C22 — Qualitative FE convergence (pp. 34, 35).** Authors say almost converged at100 per edge; no norm/tolerance, T section differences remain visible. Run and report quantitative50/100/200mesh comparison rather than treat convergence as exact.

## Limits on exact reproduction

Author-source verification or explicitly labeled reconstruction choices are necessary for α, T₀, geometry origins, exact point samples, material interface smoothing/projection, FE element formulation, optimizer details, data-loss construction, missing weight value and metric normalization. An accelerated local run can cover the study design, but it must not be described as matching manuscript budgets unless its recorded training counts do so. Figures alone do not establish random seeds or run identities.

Files: `paper_experiment_inventory.json` contains the complete structured settings and coverage; this Markdown is the readable audit.
