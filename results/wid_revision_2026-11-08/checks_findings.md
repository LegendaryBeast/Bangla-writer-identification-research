# Post-review checks (2026-11-08)

Equal writer weight within setting, equal setting weight across the 120 settings.

Post-review exploratory analyses. Stratified writer bootstrap/sign flips conditional on the enrolled gallery, fitted evidence and selected-reference score protocol. Gallery composition and fitting are not resampled.

Uniform 21/21 writer partitions; query membership and gallery memberships supplied as oracle labels. Draw quantiles measure partition variability, not sampling confidence.

Exact filter identity error: 0.
Independence approximation: MAE 1.48 points; bias -0.94 points; pooled r 0.9956.
Oracle known hand mean 57.82%; expected random balanced partition 58.60%.

                     scope                                      contrast  estimate     ci_lo     ci_hi  p_signflip  p_holm
          all_120_settings                              known_minus_none  0.061075  0.051203  0.071376      0.0001     NaN
          all_120_settings                        query_bayes_minus_none -0.005671 -0.023242  0.009207      0.5402     NaN
          all_120_settings                       query_filter_minus_none -0.142886 -0.193070 -0.095446      0.0001     NaN
          all_120_settings                         both_bayes_minus_none  0.009708  0.004510  0.015452      0.0007     NaN
          all_120_settings              known_random_expected_minus_none  0.068883  0.064389  0.073595      0.0001     NaN
          all_120_settings known_hand_minus_random_partition_expectation -0.007809 -0.017699  0.002328      0.1795     NaN
primary:resnet18:own:n1:k1                        query_bayes_minus_none -0.007937 -0.037698  0.017857      0.6957     1.0
primary:vit_b_16:own:n1:k1                        query_bayes_minus_none  0.001984 -0.025794  0.027778      1.0000     1.0
primary:resnet18:own:n4:k6                        query_bayes_minus_none -0.014881 -0.038690  0.008929      0.3664     1.0
primary:vit_b_16:own:n4:k6                        query_bayes_minus_none -0.002976 -0.032738  0.026786      1.0000     1.0

Exploratory descriptor agreement. Gallery writer-cross-fitted predictors can include the querywriter in their training cohort; no attribution to handedness specifically is established.
