# Selection check: held-out vs contaminated WildChat texts

Held-out sample: 300 pairs. Contaminated: the 259 of 300 texts of the earlier unfiltered sample (same seed) whose user turn is in Dolci; the other 41 reappear in the held-out sample.
Two-sided tests; permutation p from 20000 permutations of Welch's t.

| quantity            |   clean_mean |   clean_median |   contaminated_mean |   contaminated_median |     diff |   cohens_d |   welch_t |   perm_p |   mann_whitney_p |
|:--------------------|-------------:|---------------:|--------------------:|----------------------:|---------:|-----------:|----------:|---------:|-----------------:|
| user tokens         |       89.86  |         66     |              89.02  |                69     |  0.844   |    0.01256 |    0.1478 |   0.8786 |           0.8211 |
| assistant tokens    |      136.7   |        115     |             142.7   |               126     | -6.003   |   -0.06841 |   -0.8088 |   0.4146 |           0.322  |
| base user@plain NLL |        3.052 |          2.928 |               3.128 |                 3.175 | -0.07607 |   -0.07015 |   -0.8159 |   0.4173 |           0.2306 |
| base asst@plain NLL |        2.179 |          1.949 |               2.167 |                 1.968 |  0.01173 |    0.01359 |    0.1586 |   0.8719 |           0.3796 |
