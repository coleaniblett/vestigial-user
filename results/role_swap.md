# Role-swapped conversations

300 held-out WildChat conversations (user P1, assistant R1, user P2), each scored under two labelings that keep every content token identical:

- normal: user P1, assistant R1, user P2
- flipped: assistant P1, user R1, assistant P2

Both labelings get the default system turn. Only R1 and P2 are scored. Mean per-token NLL (nats) over conversations, 95% bootstrap CI over conversations. Label effect = NLL(flipped) - NLL(normal): positive means the turn is easier under its real role. Relative to base = the stage's label effect minus base's, paired over conversations; base measures what the flipped labeling's oddness costs on its own.

Median tokens: R1 103, P2 48, P1 58. P2 identical to P1 in 15 conversations.

| quantity                           | base                 | sft                     | dpo                     | instruct                |
|:-----------------------------------|:---------------------|:------------------------|:------------------------|:------------------------|
| NLL R1 normal                      | 1.110 [1.059, 1.159] | 1.241 [1.190, 1.294]    | 1.859 [1.788, 1.932]    | 1.889 [1.816, 1.963]    |
| NLL R1 flipped                     | 1.138 [1.089, 1.190] | 1.177 [1.124, 1.230]    | 1.570 [1.508, 1.632]    | 1.600 [1.538, 1.663]    |
| NLL P2 normal                      | 1.979 [1.838, 2.129] | 2.138 [1.987, 2.297]    | 2.423 [2.268, 2.580]    | 2.460 [2.304, 2.618]    |
| NLL P2 flipped                     | 2.024 [1.879, 2.172] | 2.289 [2.131, 2.454]    | 3.125 [2.955, 3.294]    | 3.216 [3.040, 3.394]    |
| label effect R1 (flipped - normal) | 0.029 [0.024, 0.033] | -0.064 [-0.079, -0.049] | -0.289 [-0.321, -0.259] | -0.288 [-0.322, -0.257] |
| label effect P2 (flipped - normal) | 0.045 [0.038, 0.052] | 0.151 [0.133, 0.170]    | 0.702 [0.638, 0.768]    | 0.756 [0.689, 0.827]    |
| label effect R1 relative to base   |                      | -0.093 [-0.108, -0.078] | -0.318 [-0.349, -0.287] | -0.317 [-0.350, -0.285] |
| label effect P2 relative to base   |                      | 0.106 [0.089, 0.123]    | 0.657 [0.593, 0.722]    | 0.711 [0.644, 0.780]    |

## Label effect without conversations whose P2 copies P1

P2 starting with the first 50 characters of P1 (the user resent or edited the prompt).

| stage    | turn   | excluding P2 copies (n=244)   | P2 copies only (n=56)   |
|:---------|:-------|:------------------------------|:------------------------|
| base     | R1     | 0.029 [0.024, 0.034]          | 0.028 [0.021, 0.036]    |
| base     | P2     | 0.051 [0.043, 0.060]          | 0.019 [0.012, 0.026]    |
| sft      | R1     | -0.067 [-0.085, -0.050]       | -0.051 [-0.082, -0.018] |
| sft      | P2     | 0.167 [0.146, 0.189]          | 0.081 [0.055, 0.110]    |
| dpo      | R1     | -0.277 [-0.310, -0.246]       | -0.341 [-0.430, -0.256] |
| dpo      | P2     | 0.689 [0.618, 0.763]          | 0.757 [0.603, 0.918]    |
| instruct | R1     | -0.276 [-0.310, -0.242]       | -0.342 [-0.434, -0.253] |
| instruct | P2     | 0.745 [0.671, 0.825]          | 0.802 [0.641, 0.971]    |
