# Nested fusion-weight comparison

               scope metric           contrast  estimate     ci_lo    ci_hi  p_signflip
    all_120_settings   top1  nested_minus_none -0.000815 -0.011294 0.008077      0.8784
    all_120_settings   top1 nested_minus_fixed  0.004856 -0.001467 0.012825      0.2205
    all_120_settings    nll  nested_minus_none -0.010035 -0.041018 0.023552      0.5418
    all_120_settings    nll nested_minus_fixed -0.021591 -0.054243 0.005826      0.1784
own_text_96_settings   top1  nested_minus_none -0.000767 -0.012351 0.009137      0.8966
own_text_96_settings   top1 nested_minus_fixed  0.005141 -0.002138 0.014182      0.2655
own_text_96_settings    nll  nested_minus_none -0.012860 -0.046495 0.023579      0.4764
own_text_96_settings    nll nested_minus_fixed -0.025997 -0.065540 0.006866      0.1779

     rep query  n  k           contrast  primary  estimate     ci_lo    ci_hi  p_signflip  p_holm
resnet18   own  1  1  nested_minus_none     True -0.001984 -0.021825 0.015873      1.0000     1.0
resnet18   own  1  1 nested_minus_fixed     True  0.005952 -0.009921 0.023810      0.6674     1.0
resnet18   own  4  6  nested_minus_none     True -0.008929 -0.026786 0.008929      0.5034     1.0
resnet18   own  4  6 nested_minus_fixed     True  0.005952 -0.014881 0.029762      0.7527     1.0
vit_b_16   own  1  1  nested_minus_none     True  0.011905 -0.003968 0.027778      0.2770     1.0
vit_b_16   own  1  1 nested_minus_fixed     True  0.009921 -0.005952 0.027778      0.4067     1.0
vit_b_16   own  4  6  nested_minus_none     True -0.002976 -0.029762 0.017857      1.0000     1.0
vit_b_16   own  4  6 nested_minus_fixed     True  0.000000 -0.014881 0.011905      1.0000     1.0
