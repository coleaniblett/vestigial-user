# Coin-toss spillover

40 three-turn conversations (20 items, each with heads and with tails as the preferred outcome), default system turn. A first-person preference sentence with a reason is said by the assistant (in its reply), by the user (in the opener), or by nobody; the final user turn ends "I flipped the coin and it landed on" and is left open. P(outcome) sums the probability of each case and spacing variant as the continuation (token sequences: `' heads'` -> 1 token(s); `'heads'` -> 1 token(s); `' Heads'` -> 1 token(s); `'Heads'` -> 2 token(s); `' HEADS'` -> 2 token(s); `'HEADS'` -> 2 token(s); `' tails'` -> 1 token(s); `'tails'` -> 1 token(s); `' Tails'` -> 2 token(s); `'Tails'` -> 2 token(s); `' TAILS'` -> 2 token(s); `'TAILS'` -> 2 token(s)).

Log-odds toward preferred = log P(preferred) - log P(other); positive means the model expects the outcome the stated preference favours. Because each item appears once with each outcome preferred, the 'none' mean is 0 by construction and the other means are tilts relative to it. 95% bootstrap CI over conversations. Relative to base = the stage's (assistant - user) tilt minus base's, paired over conversations. P(heads) + P(tails) is the mean probability mass on the two outcomes (how much the readout means).

| quantity                                               | base                    | sft                     | dpo                     | instruct                |
|:-------------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| log-odds toward preferred, assistant states preference | +0.292 [+0.072, +0.517] | -0.032 [-0.298, +0.243] | -0.811 [-1.251, -0.401] | -0.926 [-1.337, -0.551] |
| log-odds toward preferred, user states preference      | +0.443 [+0.239, +0.655] | +0.203 [-0.055, +0.460] | -0.690 [-1.191, -0.193] | -0.722 [-1.206, -0.259] |
| log-odds toward preferred, none states preference      | +0.000 [-0.535, +0.542] | -0.000 [-0.311, +0.314] | +0.000 [-0.221, +0.220] | -0.000 [-0.296, +0.301] |
| assistant tilt - user tilt                             | -0.151 [-0.255, -0.039] | -0.235 [-0.371, -0.095] | -0.121 [-0.339, +0.103] | -0.204 [-0.458, +0.060] |
| assistant - user tilt, relative to base                |                         | -0.084 [-0.267, +0.097] | +0.029 [-0.172, +0.233] | -0.053 [-0.279, +0.175] |
| P(heads) + P(tails), assistant                         | 0.901                   | 0.899                   | 0.787                   | 0.809                   |
| P(heads) + P(tails), user                              | 0.901                   | 0.919                   | 0.775                   | 0.802                   |
| P(heads) + P(tails), none                              | 0.887                   | 0.855                   | 0.747                   | 0.775                   |
