Random Forest Geochemical Prediction Package

Training dataset: Ref.csv (Camp Zone)
External validation: Central.csv and Crossroad.csv
Targets: Au, Ag, Cu, Zn, As, Pb
Excluded entirely: Sb, Sc
Predictors (20): Al (%), Ba (ppm), Be (ppm), Bi (ppm), Ca (%), Cd (ppm), Co (ppm), Cr (ppm), Fe (%), K (%), Mg (%), Mn (ppm), Mo (ppm), Na (%), Ni (ppm), P (ppm), S (%), Sr (ppm), Ti (%), V (ppm)
Metadata excluded from predictors: Hole_ID, Zone, coordinates, Depth_From, Depth_to
Missing predictor values: median imputation learned within each fitted pipeline; median imputation is applied.
Missing targets: never imputed; rows without a measured target are excluded for that target model.
Internal split: 80/20 grouped by Hole_ID to prevent adjacent down-hole intervals leaking across train/test.
Model specification: 150-tree Random Forest, sqrt feature sampling, min_samples_leaf=2; grouped holdout by drill hole.
Target transformation: log1p during fitting because geochemical concentrations are strongly right-skewed; predictions back-transformed to original units.
External models: tuned pipeline refitted on all available Ref measurements for each target, then frozen and applied to Central/Crossroad.
External target measurements are used only for post-prediction evaluation.
Metrics: R2, RMSE, MAE on original units plus R2 on log1p scale.
Figures: 400-dpi PNG.
Random seed: 42.
