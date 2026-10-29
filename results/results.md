# Results

Mean over texts, 95% bootstrap CI over texts. Text sources are analyzed separately and never pooled.

Texts per set: wildchat user 300, oasst1 user 300, wildchat assistant 300, olmo-instruct assistant 290.

## Raw mean NLL by text set and slot

| text set                | quantity                         | base                 | sft                  | dpo                  | instruct             |
|:------------------------|:---------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| wildchat user           | NLL user slot (user@user)        | 2.887 [2.778, 2.993] | 2.411 [2.266, 2.557] | 2.552 [2.400, 2.703] | 2.582 [2.428, 2.734] |
| wildchat user           | NLL assistant slot (user@asst)   | 2.893 [2.782, 3.003] | 2.669 [2.515, 2.823] | 2.679 [2.524, 2.825] | 2.694 [2.546, 2.847] |
| wildchat user           | NLL plain (no slot) (user@plain) | 3.052 [2.941, 3.162] | 2.848 [2.702, 2.994] | 2.958 [2.807, 3.110] | 3.011 [2.857, 3.164] |
| oasst1 user             | NLL user slot (user@user)        | 2.765 [2.690, 2.840] | 2.916 [2.839, 2.993] | 3.022 [2.938, 3.107] | 3.049 [2.964, 3.131] |
| oasst1 user             | NLL assistant slot (user@asst)   | 2.787 [2.712, 2.862] | 3.304 [3.217, 3.389] | 3.177 [3.093, 3.260] | 3.176 [3.091, 3.260] |
| oasst1 user             | NLL plain (no slot) (user@plain) | 2.970 [2.889, 3.052] | 3.245 [3.159, 3.334] | 3.369 [3.279, 3.458] | 3.423 [3.333, 3.515] |
| wildchat assistant      | NLL user slot (asst@user)        | 2.041 [1.959, 2.127] | 2.229 [2.146, 2.315] | 2.463 [2.380, 2.547] | 2.503 [2.420, 2.590] |
| wildchat assistant      | NLL assistant slot (asst@asst)   | 2.036 [1.951, 2.122] | 2.306 [2.214, 2.398] | 2.568 [2.479, 2.659] | 2.602 [2.515, 2.692] |
| wildchat assistant      | NLL plain (no slot) (asst@plain) | 2.179 [2.091, 2.274] | 2.420 [2.329, 2.514] | 2.623 [2.534, 2.716] | 2.675 [2.587, 2.766] |
| olmo-instruct assistant | NLL user slot (asst@user)        | 1.758 [1.697, 1.825] | 1.616 [1.549, 1.688] | 1.637 [1.563, 1.717] | 1.631 [1.554, 1.712] |
| olmo-instruct assistant | NLL assistant slot (asst@asst)   | 1.755 [1.693, 1.822] | 1.577 [1.503, 1.655] | 1.579 [1.501, 1.661] | 1.566 [1.488, 1.650] |
| olmo-instruct assistant | NLL plain (no slot) (asst@plain) | 1.846 [1.777, 1.916] | 1.750 [1.677, 1.828] | 1.787 [1.711, 1.866] | 1.794 [1.716, 1.878] |

## Slot effect: NLL(user slot) - NLL(assistant slot)

Default: without the first content token. When an assistant turn directly follows the system turn, the post-trained checkpoints put nearly all next-token probability on `<|im_start|>`, so the first content token in the assistant slot costs 2-14 nats whatever the text (see position.md). The all-tokens version follows for reference.

### Slot effect, excl. position 0 (default)

| text set                | base                    | sft                    | dpo                     | instruct                |
|:------------------------|:------------------------|:-----------------------|:------------------------|:------------------------|
| wildchat user           | -0.001 [-0.006, 0.003]  | -0.004 [-0.015, 0.007] | -0.041 [-0.059, -0.023] | -0.049 [-0.070, -0.029] |
| oasst1 user             | -0.023 [-0.029, -0.018] | -0.000 [-0.010, 0.009] | -0.037 [-0.053, -0.021] | -0.045 [-0.064, -0.028] |
| wildchat assistant      | 0.004 [0.001, 0.006]    | 0.087 [0.076, 0.098]   | -0.065 [-0.088, -0.043] | -0.077 [-0.101, -0.053] |
| olmo-instruct assistant | 0.002 [0.001, 0.004]    | 0.117 [0.105, 0.129]   | 0.069 [0.052, 0.086]    | 0.066 [0.048, 0.085]    |

### Slot effect, all tokens

| text set                | base                    | sft                     | dpo                     | instruct                |
|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| wildchat user           | -0.006 [-0.010, -0.002] | -0.259 [-0.283, -0.235] | -0.127 [-0.145, -0.109] | -0.113 [-0.133, -0.093] |
| oasst1 user             | -0.022 [-0.026, -0.018] | -0.389 [-0.411, -0.366] | -0.155 [-0.172, -0.138] | -0.127 [-0.145, -0.109] |
| wildchat assistant      | 0.005 [0.003, 0.008]    | -0.077 [-0.093, -0.063] | -0.105 [-0.127, -0.084] | -0.100 [-0.124, -0.076] |
| olmo-instruct assistant | 0.003 [0.001, 0.005]    | 0.039 [0.026, 0.051]    | 0.058 [0.042, 0.074]    | 0.064 [0.047, 0.082]    |

## Role-specific: NLL(own slot) - NLL(plain)

user@user - user@plain for user text; asst@asst - asst@plain for assistant text.

| text set                | quantity                               | base                    | sft                     | dpo                     | instruct                |
|:------------------------|:---------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| wildchat user           | role-specific (user@user - user@plain) | -0.165 [-0.190, -0.140] | -0.437 [-0.478, -0.399] | -0.406 [-0.448, -0.365] | -0.430 [-0.472, -0.389] |
| oasst1 user             | role-specific (user@user - user@plain) | -0.205 [-0.225, -0.185] | -0.330 [-0.357, -0.303] | -0.347 [-0.377, -0.317] | -0.374 [-0.406, -0.343] |
| wildchat assistant      | role-specific (asst@asst - asst@plain) | -0.143 [-0.164, -0.124] | -0.114 [-0.136, -0.091] | -0.055 [-0.085, -0.026] | -0.073 [-0.105, -0.042] |
| olmo-instruct assistant | role-specific (asst@asst - asst@plain) | -0.091 [-0.107, -0.076] | -0.173 [-0.194, -0.152] | -0.208 [-0.232, -0.185] | -0.228 [-0.252, -0.204] |

## Mean next-token entropy by text set and slot, excl. position 0 (default)

Entropy (nats) of the model's next-token distribution at each content token, averaged per text.

| text set                | quantity                                               | base                 | sft                  | dpo                  | instruct             |
|:------------------------|:-------------------------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| wildchat user           | entropy user slot (user@user), excl. position 0        | 2.654 [2.567, 2.742] | 1.858 [1.740, 1.976] | 1.749 [1.638, 1.862] | 1.591 [1.487, 1.700] |
| wildchat user           | entropy assistant slot (user@asst), excl. position 0   | 2.639 [2.549, 2.726] | 1.896 [1.781, 2.010] | 1.793 [1.684, 1.902] | 1.645 [1.544, 1.749] |
| wildchat user           | entropy plain (no slot) (user@plain), excl. position 0 | 2.656 [2.574, 2.740] | 2.154 [2.058, 2.248] | 2.076 [1.979, 2.171] | 1.857 [1.768, 1.947] |
| oasst1 user             | entropy user slot (user@user), excl. position 0        | 2.636 [2.575, 2.699] | 2.380 [2.319, 2.441] | 2.253 [2.193, 2.312] | 2.100 [2.042, 2.158] |
| oasst1 user             | entropy assistant slot (user@asst), excl. position 0   | 2.617 [2.554, 2.680] | 2.386 [2.323, 2.447] | 2.219 [2.159, 2.280] | 2.070 [2.011, 2.128] |
| oasst1 user             | entropy plain (no slot) (user@plain), excl. position 0 | 2.688 [2.626, 2.752] | 2.280 [2.220, 2.339] | 2.178 [2.119, 2.237] | 2.001 [1.945, 2.059] |
| wildchat assistant      | entropy user slot (asst@user), excl. position 0        | 1.912 [1.841, 1.989] | 1.755 [1.690, 1.823] | 1.590 [1.523, 1.659] | 1.454 [1.393, 1.516] |
| wildchat assistant      | entropy assistant slot (asst@asst), excl. position 0   | 1.884 [1.814, 1.959] | 1.650 [1.585, 1.715] | 1.482 [1.422, 1.544] | 1.362 [1.308, 1.418] |
| wildchat assistant      | entropy plain (no slot) (asst@plain), excl. position 0 | 2.020 [1.944, 2.097] | 1.721 [1.651, 1.794] | 1.684 [1.616, 1.756] | 1.523 [1.463, 1.586] |
| olmo-instruct assistant | entropy user slot (asst@user), excl. position 0        | 1.786 [1.731, 1.846] | 1.551 [1.500, 1.604] | 1.424 [1.372, 1.479] | 1.296 [1.249, 1.346] |
| olmo-instruct assistant | entropy assistant slot (asst@asst), excl. position 0   | 1.770 [1.714, 1.828] | 1.495 [1.445, 1.547] | 1.369 [1.319, 1.421] | 1.258 [1.214, 1.306] |
| olmo-instruct assistant | entropy plain (no slot) (asst@plain), excl. position 0 | 1.793 [1.729, 1.858] | 1.566 [1.512, 1.622] | 1.521 [1.467, 1.578] | 1.382 [1.332, 1.433] |

## Mean next-token entropy by text set and slot, all tokens

Entropy (nats) of the model's next-token distribution at each content token, averaged per text.

| text set                | quantity                                         | base                 | sft                  | dpo                  | instruct             |
|:------------------------|:-------------------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| wildchat user           | entropy user slot (user@user), all tokens        | 2.713 [2.627, 2.801] | 1.908 [1.792, 2.023] | 1.800 [1.687, 1.910] | 1.644 [1.540, 1.747] |
| wildchat user           | entropy assistant slot (user@asst), all tokens   | 2.673 [2.587, 2.764] | 1.853 [1.741, 1.962] | 1.760 [1.654, 1.868] | 1.626 [1.528, 1.725] |
| wildchat user           | entropy plain (no slot) (user@plain), all tokens | 2.740 [2.654, 2.827] | 2.239 [2.144, 2.338] | 2.158 [2.059, 2.256] | 1.936 [1.846, 2.027] |
| oasst1 user             | entropy user slot (user@user), all tokens        | 2.727 [2.665, 2.788] | 2.444 [2.385, 2.503] | 2.319 [2.259, 2.379] | 2.169 [2.111, 2.226] |
| oasst1 user             | entropy assistant slot (user@asst), all tokens   | 2.672 [2.610, 2.732] | 2.318 [2.260, 2.377] | 2.167 [2.109, 2.227] | 2.036 [1.978, 2.094] |
| oasst1 user             | entropy plain (no slot) (user@plain), all tokens | 2.814 [2.752, 2.875] | 2.405 [2.347, 2.461] | 2.298 [2.241, 2.357] | 2.118 [2.062, 2.174] |
| wildchat assistant      | entropy user slot (asst@user), all tokens        | 1.961 [1.889, 2.038] | 1.791 [1.723, 1.860] | 1.627 [1.559, 1.693] | 1.493 [1.431, 1.556] |
| wildchat assistant      | entropy assistant slot (asst@asst), all tokens   | 1.917 [1.847, 1.993] | 1.625 [1.563, 1.689] | 1.465 [1.404, 1.526] | 1.354 [1.301, 1.409] |
| wildchat assistant      | entropy plain (no slot) (asst@plain), all tokens | 2.085 [2.008, 2.165] | 1.784 [1.712, 1.859] | 1.745 [1.673, 1.818] | 1.582 [1.520, 1.646] |
| olmo-instruct assistant | entropy user slot (asst@user), all tokens        | 1.813 [1.756, 1.872] | 1.571 [1.520, 1.625] | 1.445 [1.392, 1.500] | 1.317 [1.269, 1.368] |
| olmo-instruct assistant | entropy assistant slot (asst@asst), all tokens   | 1.788 [1.734, 1.846] | 1.483 [1.434, 1.534] | 1.361 [1.312, 1.411] | 1.255 [1.212, 1.300] |
| olmo-instruct assistant | entropy plain (no slot) (asst@plain), all tokens | 1.828 [1.766, 1.892] | 1.600 [1.544, 1.657] | 1.554 [1.499, 1.611] | 1.413 [1.363, 1.468] |

## Prefilled user-turn continuations (counts of samples)

ended_turn: closed the user turn; handoff_to_assistant: then wrote the assistant header (a subset of ended_turn); assistant_register and repetition_loop: checked on the user portion only.

| stage    |   n |   ended_turn |   handoff_to_assistant |   assistant_register |   repetition_loop |
|:---------|----:|-------------:|-----------------------:|---------------------:|------------------:|
| base     |  60 |            9 |                      0 |                   12 |                12 |
| sft      |  60 |           42 |                     41 |                    0 |                 2 |
| dpo      |  60 |           37 |                     36 |                    0 |                 0 |
| instruct |  60 |           35 |                     33 |                    1 |                 1 |
