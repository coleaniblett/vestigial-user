# Results

Mean over texts, 95% bootstrap CI over texts. Text sources are analyzed separately and never pooled.

Texts per set: wildchat user 300, oasst1 user 300, wildchat assistant 300, olmo-instruct assistant 290.

Every quantity is reported without the first content token (default) and over all tokens. When an assistant turn directly follows the system turn, the post-trained checkpoints put nearly all next-token probability on `<|im_start|>`, so the first content token in the assistant slot costs 2-14 nats whatever the text (see position.md).

## Raw mean NLL by text set and slot

### excl. position 0 (default)

| text set                | quantity                                           | base                 | sft                  | dpo                  | instruct             |
|:------------------------|:---------------------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| wildchat user           | NLL user slot (user@user), excl. position 0        | 2.793 [2.686, 2.898] | 2.327 [2.185, 2.474] | 2.470 [2.321, 2.623] | 2.500 [2.349, 2.655] |
| wildchat user           | NLL assistant slot (user@asst), excl. position 0   | 2.794 [2.686, 2.902] | 2.331 [2.190, 2.475] | 2.511 [2.361, 2.660] | 2.549 [2.395, 2.697] |
| wildchat user           | NLL plain (no slot) (user@plain), excl. position 0 | 2.962 [2.854, 3.071] | 2.748 [2.603, 2.894] | 2.857 [2.704, 3.006] | 2.912 [2.760, 3.065] |
| oasst1 user             | NLL user slot (user@user), excl. position 0        | 2.671 [2.596, 2.746] | 2.849 [2.770, 2.930] | 2.958 [2.875, 3.043] | 2.986 [2.900, 3.072] |
| oasst1 user             | NLL assistant slot (user@asst), excl. position 0   | 2.694 [2.622, 2.768] | 2.850 [2.770, 2.930] | 2.995 [2.910, 3.079] | 3.031 [2.946, 3.117] |
| oasst1 user             | NLL plain (no slot) (user@plain), excl. position 0 | 2.873 [2.794, 2.955] | 3.135 [3.049, 3.223] | 3.261 [3.172, 3.352] | 3.319 [3.229, 3.412] |
| wildchat assistant      | NLL user slot (asst@user), excl. position 0        | 1.984 [1.904, 2.070] | 2.168 [2.089, 2.251] | 2.405 [2.322, 2.488] | 2.445 [2.363, 2.529] |
| wildchat assistant      | NLL assistant slot (asst@asst), excl. position 0   | 1.980 [1.900, 2.065] | 2.081 [2.001, 2.164] | 2.470 [2.383, 2.556] | 2.522 [2.434, 2.610] |
| wildchat assistant      | NLL plain (no slot) (asst@plain), excl. position 0 | 2.115 [2.028, 2.209] | 2.353 [2.265, 2.447] | 2.556 [2.468, 2.648] | 2.611 [2.521, 2.704] |
| olmo-instruct assistant | NLL user slot (asst@user), excl. position 0        | 1.721 [1.661, 1.785] | 1.574 [1.507, 1.643] | 1.595 [1.523, 1.671] | 1.588 [1.513, 1.668] |
| olmo-instruct assistant | NLL assistant slot (asst@asst), excl. position 0   | 1.718 [1.657, 1.782] | 1.457 [1.392, 1.526] | 1.526 [1.451, 1.604] | 1.522 [1.446, 1.601] |
| olmo-instruct assistant | NLL plain (no slot) (asst@plain), excl. position 0 | 1.806 [1.740, 1.877] | 1.707 [1.638, 1.780] | 1.744 [1.670, 1.822] | 1.752 [1.676, 1.832] |

### all tokens

| text set                | quantity                                     | base                 | sft                  | dpo                  | instruct             |
|:------------------------|:---------------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| wildchat user           | NLL user slot (user@user), all tokens        | 2.887 [2.777, 3.000] | 2.411 [2.264, 2.557] | 2.552 [2.397, 2.710] | 2.582 [2.429, 2.733] |
| wildchat user           | NLL assistant slot (user@asst), all tokens   | 2.893 [2.784, 3.003] | 2.669 [2.512, 2.824] | 2.679 [2.526, 2.829] | 2.694 [2.543, 2.845] |
| wildchat user           | NLL plain (no slot) (user@plain), all tokens | 3.052 [2.943, 3.166] | 2.848 [2.702, 2.995] | 2.958 [2.810, 3.108] | 3.011 [2.859, 3.162] |
| oasst1 user             | NLL user slot (user@user), all tokens        | 2.765 [2.691, 2.840] | 2.916 [2.837, 2.997] | 3.022 [2.938, 3.104] | 3.049 [2.964, 3.133] |
| oasst1 user             | NLL assistant slot (user@asst), all tokens   | 2.787 [2.714, 2.860] | 3.304 [3.217, 3.389] | 3.177 [3.091, 3.266] | 3.176 [3.091, 3.261] |
| oasst1 user             | NLL plain (no slot) (user@plain), all tokens | 2.970 [2.890, 3.050] | 3.245 [3.157, 3.331] | 3.369 [3.281, 3.459] | 3.423 [3.334, 3.514] |
| wildchat assistant      | NLL user slot (asst@user), all tokens        | 2.041 [1.960, 2.125] | 2.229 [2.148, 2.313] | 2.463 [2.378, 2.548] | 2.503 [2.420, 2.588] |
| wildchat assistant      | NLL assistant slot (asst@asst), all tokens   | 2.036 [1.955, 2.119] | 2.306 [2.214, 2.397] | 2.568 [2.480, 2.661] | 2.602 [2.513, 2.692] |
| wildchat assistant      | NLL plain (no slot) (asst@plain), all tokens | 2.179 [2.092, 2.271] | 2.420 [2.329, 2.513] | 2.623 [2.533, 2.719] | 2.675 [2.585, 2.769] |
| olmo-instruct assistant | NLL user slot (asst@user), all tokens        | 1.758 [1.696, 1.823] | 1.616 [1.548, 1.688] | 1.637 [1.562, 1.717] | 1.631 [1.554, 1.711] |
| olmo-instruct assistant | NLL assistant slot (asst@asst), all tokens   | 1.755 [1.693, 1.824] | 1.577 [1.502, 1.656] | 1.579 [1.502, 1.661] | 1.566 [1.486, 1.650] |
| olmo-instruct assistant | NLL plain (no slot) (asst@plain), all tokens | 1.846 [1.777, 1.920] | 1.750 [1.676, 1.826] | 1.787 [1.711, 1.868] | 1.794 [1.715, 1.876] |


## Slot effect: NLL(user slot) - NLL(assistant slot)

### Slot effect, excl. position 0 (default)

| text set                | base                    | sft                    | dpo                     | instruct                |
|:------------------------|:------------------------|:-----------------------|:------------------------|:------------------------|
| wildchat user           | -0.001 [-0.006, 0.003]  | -0.004 [-0.015, 0.007] | -0.041 [-0.058, -0.023] | -0.049 [-0.069, -0.029] |
| oasst1 user             | -0.023 [-0.029, -0.018] | -0.000 [-0.010, 0.009] | -0.037 [-0.052, -0.022] | -0.045 [-0.063, -0.028] |
| wildchat assistant      | 0.004 [0.001, 0.006]    | 0.087 [0.076, 0.098]   | -0.065 [-0.088, -0.043] | -0.077 [-0.101, -0.054] |
| olmo-instruct assistant | 0.002 [0.001, 0.004]    | 0.117 [0.105, 0.129]   | 0.069 [0.052, 0.086]    | 0.066 [0.049, 0.084]    |

### Slot effect, all tokens

| text set                | base                    | sft                     | dpo                     | instruct                |
|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| wildchat user           | -0.006 [-0.010, -0.002] | -0.259 [-0.282, -0.236] | -0.127 [-0.145, -0.109] | -0.113 [-0.132, -0.093] |
| oasst1 user             | -0.022 [-0.026, -0.018] | -0.389 [-0.412, -0.365] | -0.155 [-0.172, -0.138] | -0.127 [-0.145, -0.109] |
| wildchat assistant      | 0.005 [0.003, 0.008]    | -0.077 [-0.093, -0.063] | -0.105 [-0.127, -0.084] | -0.100 [-0.123, -0.077] |
| olmo-instruct assistant | 0.003 [0.001, 0.005]    | 0.039 [0.026, 0.052]    | 0.058 [0.041, 0.074]    | 0.064 [0.047, 0.081]    |


## Role-specific: NLL(own slot) - NLL(plain)

user@user - user@plain for user text; asst@asst - asst@plain for assistant text.

### excl. position 0 (default)

| text set                | quantity                                                 | base                    | sft                     | dpo                     | instruct                |
|:------------------------|:---------------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| wildchat user           | role-specific (user@user - user@plain), excl. position 0 | -0.169 [-0.196, -0.144] | -0.421 [-0.461, -0.382] | -0.387 [-0.429, -0.345] | -0.412 [-0.455, -0.372] |
| oasst1 user             | role-specific (user@user - user@plain), excl. position 0 | -0.202 [-0.223, -0.181] | -0.285 [-0.313, -0.258] | -0.303 [-0.333, -0.274] | -0.333 [-0.365, -0.301] |
| wildchat assistant      | role-specific (asst@asst - asst@plain), excl. position 0 | -0.135 [-0.156, -0.116] | -0.272 [-0.298, -0.247] | -0.086 [-0.119, -0.055] | -0.089 [-0.122, -0.057] |
| olmo-instruct assistant | role-specific (asst@asst - asst@plain), excl. position 0 | -0.088 [-0.105, -0.072] | -0.250 [-0.273, -0.229] | -0.219 [-0.243, -0.195] | -0.230 [-0.255, -0.206] |

### all tokens

| text set                | quantity                                           | base                    | sft                     | dpo                     | instruct                |
|:------------------------|:---------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| wildchat user           | role-specific (user@user - user@plain), all tokens | -0.165 [-0.189, -0.141] | -0.437 [-0.476, -0.399] | -0.406 [-0.449, -0.366] | -0.430 [-0.471, -0.389] |
| oasst1 user             | role-specific (user@user - user@plain), all tokens | -0.205 [-0.225, -0.184] | -0.330 [-0.357, -0.303] | -0.347 [-0.377, -0.318] | -0.374 [-0.406, -0.342] |
| wildchat assistant      | role-specific (asst@asst - asst@plain), all tokens | -0.143 [-0.163, -0.124] | -0.114 [-0.137, -0.091] | -0.055 [-0.084, -0.026] | -0.073 [-0.105, -0.042] |
| olmo-instruct assistant | role-specific (asst@asst - asst@plain), all tokens | -0.091 [-0.107, -0.076] | -0.173 [-0.195, -0.153] | -0.208 [-0.232, -0.184] | -0.228 [-0.253, -0.204] |


## Mean next-token entropy by text set and slot, excl. position 0 (default)

Entropy (nats) of the model's next-token distribution at each content token, averaged per text.

| text set                | quantity                                               | base                 | sft                  | dpo                  | instruct             |
|:------------------------|:-------------------------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| wildchat user           | entropy user slot (user@user), excl. position 0        | 2.654 [2.567, 2.744] | 1.858 [1.745, 1.976] | 1.749 [1.638, 1.861] | 1.591 [1.488, 1.698] |
| wildchat user           | entropy assistant slot (user@asst), excl. position 0   | 2.639 [2.550, 2.726] | 1.896 [1.782, 2.012] | 1.793 [1.682, 1.901] | 1.645 [1.541, 1.747] |
| wildchat user           | entropy plain (no slot) (user@plain), excl. position 0 | 2.656 [2.573, 2.740] | 2.154 [2.061, 2.251] | 2.076 [1.980, 2.172] | 1.857 [1.766, 1.946] |
| oasst1 user             | entropy user slot (user@user), excl. position 0        | 2.636 [2.573, 2.699] | 2.380 [2.319, 2.442] | 2.253 [2.193, 2.313] | 2.100 [2.040, 2.158] |
| oasst1 user             | entropy assistant slot (user@asst), excl. position 0   | 2.617 [2.553, 2.681] | 2.386 [2.325, 2.446] | 2.219 [2.159, 2.280] | 2.070 [2.011, 2.129] |
| oasst1 user             | entropy plain (no slot) (user@plain), excl. position 0 | 2.688 [2.625, 2.751] | 2.280 [2.221, 2.338] | 2.178 [2.120, 2.238] | 2.001 [1.946, 2.058] |
| wildchat assistant      | entropy user slot (asst@user), excl. position 0        | 1.912 [1.839, 1.986] | 1.755 [1.689, 1.823] | 1.590 [1.525, 1.658] | 1.454 [1.394, 1.516] |
| wildchat assistant      | entropy assistant slot (asst@asst), excl. position 0   | 1.884 [1.812, 1.957] | 1.650 [1.586, 1.719] | 1.482 [1.421, 1.544] | 1.362 [1.308, 1.419] |
| wildchat assistant      | entropy plain (no slot) (asst@plain), excl. position 0 | 2.020 [1.945, 2.097] | 1.721 [1.651, 1.793] | 1.684 [1.614, 1.755] | 1.523 [1.463, 1.585] |
| olmo-instruct assistant | entropy user slot (asst@user), excl. position 0        | 1.786 [1.731, 1.847] | 1.551 [1.501, 1.604] | 1.424 [1.373, 1.478] | 1.296 [1.249, 1.346] |
| olmo-instruct assistant | entropy assistant slot (asst@asst), excl. position 0   | 1.770 [1.715, 1.827] | 1.495 [1.445, 1.548] | 1.369 [1.319, 1.421] | 1.258 [1.213, 1.306] |
| olmo-instruct assistant | entropy plain (no slot) (asst@plain), excl. position 0 | 1.793 [1.731, 1.856] | 1.566 [1.514, 1.620] | 1.521 [1.467, 1.578] | 1.382 [1.333, 1.434] |

## Mean next-token entropy by text set and slot, all tokens

Entropy (nats) of the model's next-token distribution at each content token, averaged per text.

| text set                | quantity                                         | base                 | sft                  | dpo                  | instruct             |
|:------------------------|:-------------------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| wildchat user           | entropy user slot (user@user), all tokens        | 2.713 [2.623, 2.800] | 1.908 [1.791, 2.023] | 1.800 [1.689, 1.912] | 1.644 [1.538, 1.749] |
| wildchat user           | entropy assistant slot (user@asst), all tokens   | 2.673 [2.587, 2.760] | 1.853 [1.741, 1.965] | 1.760 [1.652, 1.865] | 1.626 [1.530, 1.725] |
| wildchat user           | entropy plain (no slot) (user@plain), all tokens | 2.740 [2.655, 2.826] | 2.239 [2.144, 2.338] | 2.158 [2.058, 2.256] | 1.936 [1.846, 2.026] |
| oasst1 user             | entropy user slot (user@user), all tokens        | 2.727 [2.665, 2.789] | 2.444 [2.385, 2.503] | 2.319 [2.260, 2.376] | 2.169 [2.111, 2.225] |
| oasst1 user             | entropy assistant slot (user@asst), all tokens   | 2.672 [2.609, 2.733] | 2.318 [2.259, 2.376] | 2.167 [2.107, 2.225] | 2.036 [1.980, 2.094] |
| oasst1 user             | entropy plain (no slot) (user@plain), all tokens | 2.814 [2.751, 2.875] | 2.405 [2.348, 2.462] | 2.298 [2.238, 2.358] | 2.118 [2.063, 2.174] |
| wildchat assistant      | entropy user slot (asst@user), all tokens        | 1.961 [1.889, 2.039] | 1.791 [1.722, 1.858] | 1.627 [1.558, 1.696] | 1.493 [1.433, 1.557] |
| wildchat assistant      | entropy assistant slot (asst@asst), all tokens   | 1.917 [1.844, 1.991] | 1.625 [1.562, 1.688] | 1.465 [1.406, 1.527] | 1.354 [1.300, 1.409] |
| wildchat assistant      | entropy plain (no slot) (asst@plain), all tokens | 2.085 [2.005, 2.165] | 1.784 [1.711, 1.858] | 1.745 [1.675, 1.820] | 1.582 [1.518, 1.647] |
| olmo-instruct assistant | entropy user slot (asst@user), all tokens        | 1.813 [1.757, 1.872] | 1.571 [1.520, 1.627] | 1.445 [1.392, 1.499] | 1.317 [1.270, 1.368] |
| olmo-instruct assistant | entropy assistant slot (asst@asst), all tokens   | 1.788 [1.732, 1.848] | 1.483 [1.435, 1.534] | 1.361 [1.314, 1.410] | 1.255 [1.211, 1.301] |
| olmo-instruct assistant | entropy plain (no slot) (asst@plain), all tokens | 1.828 [1.764, 1.895] | 1.600 [1.545, 1.658] | 1.554 [1.498, 1.613] | 1.413 [1.364, 1.467] |

## Prefilled user-turn continuations (counts of samples)

ended_turn: closed the user turn; handoff_to_assistant: then wrote the assistant header (a subset of ended_turn); assistant_register and repetition_loop: checked on the user portion only.

| stage    |   n |   ended_turn |   handoff_to_assistant |   assistant_register |   repetition_loop |
|:---------|----:|-------------:|-----------------------:|---------------------:|------------------:|
| base     |  60 |            9 |                      0 |                   12 |                12 |
| sft      |  60 |           42 |                     41 |                    0 |                 2 |
| dpo      |  60 |           37 |                     36 |                    0 |                 0 |
| instruct |  60 |           35 |                     33 |                    1 |                 1 |
