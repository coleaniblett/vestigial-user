# Results

Mean over texts, 95% bootstrap CI over texts. Text sources are analyzed separately and never pooled.

Texts per set: wildchat user 300, oasst1 user 300, wildchat assistant 300, olmo-instruct assistant 290.

## Raw mean NLL by text set and slot

| text set                | quantity                         | base                 | sft                  | dpo                  | instruct             |
|:------------------------|:---------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| wildchat user           | NLL user slot (user@user)        | 2.860 [2.752, 2.966] | 2.395 [2.248, 2.538] | 2.540 [2.388, 2.690] | 2.569 [2.419, 2.719] |
| wildchat user           | NLL assistant slot (user@asst)   | 2.923 [2.812, 3.034] | 2.542 [2.401, 2.683] | 2.859 [2.706, 3.008] | 2.942 [2.788, 3.094] |
| wildchat user           | NLL plain (no slot) (user@plain) | 3.052 [2.941, 3.162] | 2.848 [2.701, 2.995] | 2.958 [2.809, 3.110] | 3.011 [2.861, 3.163] |
| oasst1 user             | NLL user slot (user@user)        | 2.712 [2.640, 2.786] | 2.849 [2.769, 2.929] | 2.947 [2.865, 3.029] | 2.969 [2.884, 3.052] |
| oasst1 user             | NLL assistant slot (user@asst)   | 2.804 [2.730, 2.880] | 3.007 [2.926, 3.088] | 3.354 [3.266, 3.438] | 3.444 [3.358, 3.533] |
| oasst1 user             | NLL plain (no slot) (user@plain) | 2.970 [2.890, 3.050] | 3.245 [3.160, 3.332] | 3.369 [3.281, 3.456] | 3.423 [3.333, 3.516] |
| wildchat assistant      | NLL user slot (asst@user)        | 2.052 [1.971, 2.135] | 2.200 [2.117, 2.286] | 2.476 [2.393, 2.560] | 2.515 [2.433, 2.599] |
| wildchat assistant      | NLL assistant slot (asst@asst)   | 2.046 [1.962, 2.133] | 2.182 [2.099, 2.268] | 2.628 [2.538, 2.723] | 2.710 [2.615, 2.809] |
| wildchat assistant      | NLL plain (no slot) (asst@plain) | 2.179 [2.089, 2.274] | 2.420 [2.330, 2.514] | 2.623 [2.533, 2.714] | 2.675 [2.585, 2.772] |
| olmo-instruct assistant | NLL user slot (asst@user)        | 1.770 [1.706, 1.839] | 1.586 [1.519, 1.658] | 1.613 [1.538, 1.693] | 1.608 [1.532, 1.687] |
| olmo-instruct assistant | NLL assistant slot (asst@asst)   | 1.766 [1.704, 1.833] | 1.556 [1.488, 1.630] | 1.698 [1.613, 1.786] | 1.720 [1.632, 1.813] |
| olmo-instruct assistant | NLL plain (no slot) (asst@plain) | 1.846 [1.777, 1.918] | 1.750 [1.678, 1.827] | 1.787 [1.712, 1.868] | 1.794 [1.715, 1.877] |

## Slot effect: NLL(user slot) - NLL(assistant slot)

| text set                | base                    | sft                     | dpo                     | instruct                |
|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| wildchat user           | -0.063 [-0.071, -0.056] | -0.147 [-0.167, -0.128] | -0.318 [-0.352, -0.288] | -0.373 [-0.410, -0.337] |
| oasst1 user             | -0.092 [-0.099, -0.084] | -0.158 [-0.175, -0.142] | -0.406 [-0.440, -0.373] | -0.475 [-0.513, -0.439] |
| wildchat assistant      | 0.006 [0.002, 0.011]    | 0.018 [0.007, 0.028]    | -0.152 [-0.176, -0.130] | -0.195 [-0.223, -0.168] |
| olmo-instruct assistant | 0.004 [0.001, 0.007]    | 0.030 [0.019, 0.040]    | -0.085 [-0.105, -0.067] | -0.113 [-0.135, -0.092] |

## Role-specific: NLL(own slot) - NLL(plain)

user@user - user@plain for user text; asst@asst - asst@plain for assistant text.

| text set                | quantity                               | base                    | sft                     | dpo                     | instruct                |
|:------------------------|:---------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| wildchat user           | role-specific (user@user - user@plain) | -0.191 [-0.217, -0.167] | -0.453 [-0.491, -0.416] | -0.418 [-0.458, -0.379] | -0.443 [-0.482, -0.404] |
| oasst1 user             | role-specific (user@user - user@plain) | -0.257 [-0.280, -0.235] | -0.396 [-0.425, -0.369] | -0.422 [-0.453, -0.391] | -0.454 [-0.487, -0.422] |
| wildchat assistant      | role-specific (asst@asst - asst@plain) | -0.133 [-0.153, -0.113] | -0.237 [-0.263, -0.213] | 0.005 [-0.025, 0.034]   | 0.035 [0.004, 0.067]    |
| olmo-instruct assistant | role-specific (asst@asst - asst@plain) | -0.080 [-0.096, -0.065] | -0.194 [-0.217, -0.172] | -0.089 [-0.117, -0.061] | -0.074 [-0.104, -0.044] |

## Prefilled user-turn continuations (counts of samples)

ended_turn: closed the user turn; handoff_to_assistant: then wrote the assistant header (a subset of ended_turn); assistant_register and repetition_loop: checked on the user portion only.

| stage    |   n |   ended_turn |   handoff_to_assistant |   assistant_register |   repetition_loop |
|:---------|----:|-------------:|-----------------------:|---------------------:|------------------:|
| base     |  60 |            4 |                      0 |                   15 |                 7 |
| sft      |  60 |           48 |                     48 |                    0 |                 0 |
| dpo      |  60 |           45 |                     44 |                    0 |                 0 |
| instruct |  60 |           44 |                     42 |                    0 |                 0 |
