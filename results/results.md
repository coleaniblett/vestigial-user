# Results

Mean over texts, 95% bootstrap CI over texts. Text sources are analyzed separately and never pooled.

Texts per set: wildchat user 300, oasst1 user 300, wildchat assistant 300, olmo-instruct assistant 290.

## Raw mean NLL by text set and slot

| text set                | quantity                         | base                 | sft                  | dpo                  | instruct             |
|:------------------------|:---------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| wildchat user           | NLL user slot (user@user)        | 2.887 [2.778, 2.993] | 2.411 [2.263, 2.558] | 2.552 [2.396, 2.705] | 2.582 [2.429, 2.736] |
| wildchat user           | NLL assistant slot (user@asst)   | 2.893 [2.782, 3.003] | 2.669 [2.517, 2.824] | 2.679 [2.525, 2.828] | 2.694 [2.544, 2.845] |
| wildchat user           | NLL plain (no slot) (user@plain) | 3.052 [2.941, 3.162] | 2.848 [2.701, 2.995] | 2.958 [2.809, 3.110] | 3.011 [2.861, 3.163] |
| oasst1 user             | NLL user slot (user@user)        | 2.765 [2.692, 2.840] | 2.916 [2.836, 2.995] | 3.022 [2.939, 3.104] | 3.049 [2.964, 3.132] |
| oasst1 user             | NLL assistant slot (user@asst)   | 2.787 [2.713, 2.861] | 3.304 [3.219, 3.390] | 3.177 [3.093, 3.259] | 3.176 [3.091, 3.262] |
| oasst1 user             | NLL plain (no slot) (user@plain) | 2.970 [2.890, 3.050] | 3.245 [3.160, 3.332] | 3.369 [3.281, 3.456] | 3.423 [3.333, 3.516] |
| wildchat assistant      | NLL user slot (asst@user)        | 2.041 [1.959, 2.124] | 2.229 [2.146, 2.315] | 2.463 [2.380, 2.547] | 2.503 [2.420, 2.588] |
| wildchat assistant      | NLL assistant slot (asst@asst)   | 2.036 [1.953, 2.121] | 2.306 [2.217, 2.399] | 2.568 [2.483, 2.659] | 2.602 [2.513, 2.693] |
| wildchat assistant      | NLL plain (no slot) (asst@plain) | 2.179 [2.089, 2.274] | 2.420 [2.330, 2.514] | 2.623 [2.533, 2.714] | 2.675 [2.585, 2.772] |
| olmo-instruct assistant | NLL user slot (asst@user)        | 1.758 [1.694, 1.827] | 1.616 [1.548, 1.688] | 1.637 [1.562, 1.716] | 1.631 [1.555, 1.709] |
| olmo-instruct assistant | NLL assistant slot (asst@asst)   | 1.755 [1.693, 1.821] | 1.577 [1.505, 1.655] | 1.579 [1.501, 1.660] | 1.566 [1.488, 1.649] |
| olmo-instruct assistant | NLL plain (no slot) (asst@plain) | 1.846 [1.777, 1.918] | 1.750 [1.678, 1.827] | 1.787 [1.712, 1.868] | 1.794 [1.715, 1.877] |

## Slot effect: NLL(user slot) - NLL(assistant slot)

| text set                | base                    | sft                     | dpo                     | instruct                |
|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| wildchat user           | -0.006 [-0.010, -0.002] | -0.259 [-0.282, -0.236] | -0.127 [-0.145, -0.109] | -0.113 [-0.133, -0.093] |
| oasst1 user             | -0.022 [-0.026, -0.018] | -0.389 [-0.412, -0.366] | -0.155 [-0.172, -0.138] | -0.127 [-0.145, -0.109] |
| wildchat assistant      | 0.005 [0.003, 0.008]    | -0.077 [-0.093, -0.062] | -0.105 [-0.126, -0.084] | -0.100 [-0.123, -0.077] |
| olmo-instruct assistant | 0.003 [0.001, 0.005]    | 0.039 [0.026, 0.052]    | 0.058 [0.041, 0.074]    | 0.064 [0.048, 0.082]    |

## Role-specific: NLL(own slot) - NLL(plain)

user@user - user@plain for user text; asst@asst - asst@plain for assistant text.

| text set                | quantity                               | base                    | sft                     | dpo                     | instruct                |
|:------------------------|:---------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| wildchat user           | role-specific (user@user - user@plain) | -0.165 [-0.190, -0.140] | -0.437 [-0.477, -0.399] | -0.406 [-0.449, -0.365] | -0.430 [-0.472, -0.389] |
| oasst1 user             | role-specific (user@user - user@plain) | -0.205 [-0.225, -0.184] | -0.330 [-0.357, -0.303] | -0.347 [-0.378, -0.318] | -0.374 [-0.406, -0.343] |
| wildchat assistant      | role-specific (asst@asst - asst@plain) | -0.143 [-0.164, -0.124] | -0.114 [-0.137, -0.091] | -0.055 [-0.085, -0.027] | -0.073 [-0.104, -0.042] |
| olmo-instruct assistant | role-specific (asst@asst - asst@plain) | -0.091 [-0.107, -0.076] | -0.173 [-0.194, -0.152] | -0.208 [-0.232, -0.185] | -0.228 [-0.252, -0.205] |

## Prefilled user-turn continuations (counts of samples)

ended_turn: closed the user turn; handoff_to_assistant: then wrote the assistant header (a subset of ended_turn); assistant_register and repetition_loop: checked on the user portion only.

| stage    |   n |   ended_turn |   handoff_to_assistant |   assistant_register |   repetition_loop |
|:---------|----:|-------------:|-----------------------:|---------------------:|------------------:|
| base     |  60 |            9 |                      0 |                   12 |                12 |
| sft      |  60 |           42 |                     41 |                    0 |                 2 |
| dpo      |  60 |           37 |                     36 |                    0 |                 0 |
| instruct |  60 |           35 |                     33 |                    1 |                 1 |
