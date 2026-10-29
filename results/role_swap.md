# Role-swapped conversations

300 held-out WildChat conversations (user P1, assistant R1, user P2), each scored under two labelings that keep every content token identical:

- normal: user P1, assistant R1, user P2
- flipped: assistant P1, user R1, assistant P2

Both labelings get the default system turn. Mean per-token NLL (nats) over conversations, 95% bootstrap CI over conversations. Label effect = NLL(flipped) - NLL(normal): positive means the turn is easier under its real role. Relative to base = the stage's label effect minus base's, paired over conversations; base measures what the flipped labeling's oddness costs on its own.

## Three-turn conversations (P1, R1, P2), ChatGPT's R1; R1 and P2 scored

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

### Label effect without conversations whose P2 copies P1

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

## Two-turn conversations (P1, R1): ChatGPT's reply vs OLMo Instruct's own reply

Normal: user P1, assistant R1. Flipped: assistant P1, user R1. R1 is either ChatGPT's original reply or OLMo 3 7B Instruct's greedy reply to P1 (max 300 new tokens; cut replies kept). Only conversations with both replies are used (n=294; 6 OLMo replies failed the length/render filters; 136 of the kept OLMo replies are cut at 300 tokens). Interaction = OLMo label effect - ChatGPT label effect, paired over conversations.

**Circularity at the Instruct stage:** the OLMo replies were generated greedily by the Instruct checkpoint itself in the normal two-turn context, so at that stage their normal-label NLL is low by construction and the Instruct-stage OLMo label effect and interaction are partly mechanical. SFT and DPO score the same text without having generated it.

| quantity                                                    | base                    | sft                     | dpo                     | instruct                |
|:------------------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| 2turn NLL R1 chatgpt normal                                 | 1.107 [1.057, 1.157]    | 1.238 [1.186, 1.291]    | 1.859 [1.788, 1.933]    | 1.887 [1.814, 1.960]    |
| 2turn NLL R1 chatgpt flipped                                | 1.136 [1.085, 1.185]    | 1.172 [1.119, 1.226]    | 1.566 [1.504, 1.626]    | 1.595 [1.534, 1.657]    |
| 2turn NLL R1 olmo-instruct normal                           | 1.109 [1.075, 1.143]    | 0.520 [0.503, 0.537]    | 0.402 [0.386, 0.419]    | 0.350 [0.336, 0.365]    |
| 2turn NLL R1 olmo-instruct flipped                          | 1.128 [1.093, 1.162]    | 0.671 [0.646, 0.698]    | 0.509 [0.487, 0.531]    | 0.467 [0.446, 0.489]    |
| 2turn label effect R1 chatgpt                               | 0.029 [0.024, 0.033]    | -0.066 [-0.081, -0.051] | -0.293 [-0.324, -0.263] | -0.292 [-0.325, -0.260] |
| 2turn label effect R1 olmo-instruct                         | 0.019 [0.015, 0.023]    | 0.151 [0.133, 0.170]    | 0.106 [0.091, 0.122]    | 0.117 [0.102, 0.134]    |
| 2turn interaction: OLMo label effect - ChatGPT label effect | -0.010 [-0.015, -0.006] | 0.217 [0.196, 0.240]    | 0.400 [0.367, 0.433]    | 0.409 [0.374, 0.445]    |
| 2turn label effect R1 chatgpt relative to base              |                         | -0.095 [-0.109, -0.080] | -0.322 [-0.354, -0.291] | -0.321 [-0.354, -0.287] |
| 2turn label effect R1 olmo-instruct relative to base        |                         | 0.133 [0.116, 0.150]    | 0.087 [0.074, 0.103]    | 0.098 [0.085, 0.113]    |
| 2turn interaction relative to base                          |                         | 0.227 [0.207, 0.249]    | 0.410 [0.377, 0.443]    | 0.419 [0.384, 0.455]    |

## Mean next-token entropy over the scored content tokens (nats)

Entropy of the model's next-token distribution at each content token of the scored turn, averaged per conversation; 95% bootstrap CI over conversations.

| quantity                                  | base                 | sft                  | dpo                  | instruct             |
|:------------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| entropy 3turn R1, normal                  | 1.161 [1.115, 1.208] | 0.878 [0.836, 0.921] | 0.835 [0.791, 0.879] | 0.768 [0.727, 0.807] |
| entropy 3turn R1, flipped                 | 1.216 [1.167, 1.262] | 1.004 [0.962, 1.045] | 0.902 [0.863, 0.942] | 0.832 [0.795, 0.869] |
| entropy 3turn P2, normal                  | 1.957 [1.828, 2.088] | 1.765 [1.651, 1.883] | 1.739 [1.639, 1.837] | 1.599 [1.506, 1.694] |
| entropy 3turn P2, flipped                 | 1.923 [1.796, 2.053] | 1.679 [1.561, 1.801] | 1.559 [1.464, 1.659] | 1.411 [1.324, 1.500] |
| entropy 2turn R1 = ChatGPT reply, normal  | 1.158 [1.111, 1.204] | 0.878 [0.834, 0.923] | 0.835 [0.790, 0.880] | 0.767 [0.726, 0.809] |
| entropy 2turn R1 = ChatGPT reply, flipped | 1.213 [1.165, 1.260] | 0.999 [0.956, 1.040] | 0.899 [0.858, 0.941] | 0.829 [0.792, 0.867] |
| entropy 2turn R1 = OLMo reply, normal     | 1.244 [1.202, 1.284] | 0.956 [0.915, 0.996] | 0.876 [0.836, 0.917] | 0.803 [0.768, 0.840] |
| entropy 2turn R1 = OLMo reply, flipped    | 1.276 [1.234, 1.318] | 1.010 [0.974, 1.047] | 0.911 [0.874, 0.950] | 0.834 [0.801, 0.867] |
