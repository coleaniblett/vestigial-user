# Model checks

| stage | repo | sha | license | gated | weights (GB) | vocab | chat template sha | tokenizer.json sha |
|---|---|---|---|---|---|---|---|---|
| base | `allenai/Olmo-3-1025-7B` | a81bae42db | apache-2.0 | False | 14.6 | 100278 | None | 73fd5254624f |
| sft | `allenai/Olmo-3-7B-Instruct-SFT` | e1452fc572 | apache-2.0 | False | 14.6 | 100278 | edbb35ef1219 | 32a022d09976 |
| dpo | `allenai/Olmo-3-7B-Instruct-DPO` | b33130b7de | apache-2.0 | False | 14.6 | 100278 | edbb35ef1219 | 7738a25c46a6 |
| instruct | `allenai/Olmo-3-7B-Instruct` | 6e5971d9eb | apache-2.0 | False | 14.6 | 100278 | f5186d42d99c | 7738a25c46a6 |

## Chat templates

Post-trained checkpoints share one chat template file: **False**.
Post-trained templates render a plain user/assistant exchange identically: **True**.

Template file of sft, dpo vs instruct (unified diff):

```diff
@@ -2 +2 @@
-You are a helpful function-calling AI assistant. ' -}}{%- if tools is none -%}{{- 'You do not currently have access to any functions. <functions></functions><|im_end|>
+You are a helpful function-calling AI assistant. ' -}}{%- if tools is none or (tools | length) == 0 -%}{{- 'You do not currently have access to any functions. <functions></functions><|im_end|>
```

Base has its own chat template: False; it is scored with the Instruct template.

## Tokenizers

All four share one tokenizer.json file: **False**.
All four share one merge list: **True**.
All four share one vocab: **False**.

`base` vocab entries differing from Instruct (token, base id, instruct id):

- `</function_calls>`: None vs 100269
- `</functions>`: None vs 100267
- `<function_calls>`: None vs 100268
- `<functions>`: None vs 100266
- `<|extra_id_10|>`: 100275 vs None
- `<|extra_id_1|>`: 100266 vs 100270
- `<|extra_id_2|>`: 100267 vs 100271
- `<|extra_id_3|>`: 100268 vs 100272
- `<|extra_id_4|>`: 100269 vs 100273
- `<|extra_id_5|>`: 100270 vs 100274
- `<|extra_id_6|>`: 100271 vs 100275
- `<|extra_id_7|>`: 100272 vs None
- `<|extra_id_8|>`: 100273 vs None
- `<|extra_id_9|>`: 100274 vs None

Vocab differences are confined to reserved/added tokens: **True**.

Special tokens: bos='<|endoftext|>' eos='<|endoftext|>' pad='<|pad|>'

Instruct template rendering of a user + assistant exchange:

```
<|im_start|>system
You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>
<|im_start|>user
USER<|im_end|>
<|im_start|>assistant
ASSISTANT<|endoftext|>
```

Instruct chat template:

```jinja
{%- set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 -%}{%- if not has_system -%}{{- '<|im_start|>system
You are a helpful function-calling AI assistant. ' -}}{%- if tools is none -%}{{- 'You do not currently have access to any functions. <functions></functions><|im_end|>
' -}}{%- else -%}{{- 'You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Do not make assumptions about what values to plug into functions.' -}}{{- '<functions>' -}}{{- tools | tojson -}}{{- '</functions><|im_end|>
' -}}{%- endif -%}{%- endif -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{{- '<|im_start|>system
' + message['content'] -}}{%- if tools is not none -%}{{- '<functions>' -}}{{- tools | tojson -}}{{- '</functions>' -}}{%- elif message.get('functions', none) is not none -%}{{- ' <functions>' + message['functions'] + '</functions>' -}}{%- endif -%}{{- '<|im_end|>
' -}}{%- elif message['role'] == 'user' -%}{{- '<|im_start|>user
' + message['content'] + '<|im_end|>
' -}}{%- elif message['role'] == 'assistant' -%}{{- '<|im_start|>assistant
' -}}{%- if message.get('content', none) is not none -%}{{- message['content'] -}}{%- endif -%}{%- if message.get('function_calls', none) is not none -%}{{- '<function_calls>' + message['function_calls'] + '</function_calls>' -}}{% elif message.get('tool_calls', none) is not none %}{{- '<function_calls>' -}}{%- for tool_call in message['tool_calls'] %}{%- if tool_call is mapping and tool_call.get('function', none) is not none %}{%- set args = tool_call['function']['arguments'] -%}{%- set ns = namespace(arguments_list=[]) -%}{%- for key, value in args.items() -%}{%- set ns.arguments_list = ns.arguments_list + [key ~ '=' ~ (value | tojson)] -%}{%- endfor -%}{%- set arguments = ns.arguments_list | join(', ') -%}{{- tool_call['function']['name'] + '(' + arguments + ')' -}}{%- if not loop.last -%}{{ '
' }}{%- endif -%}{% else %}{{- tool_call -}}{%- endif %}{%- endfor %}{{- '</function_calls>' -}}{%- endif -%}{%- if not loop.last -%}{{- '<|im_end|>' + '
' -}}{%- else -%}{{- eos_token -}}{%- endif -%}{%- elif message['role'] == 'environment' -%}{{- '<|im_start|>environment
' + message['content'] + '<|im_end|>
' -}}{%- elif message['role'] == 'tool' -%}{{- '<|im_start|>environment
' + message['content'] + '<|im_end|>
' -}}{%- endif -%}{%- if loop.last and add_generation_prompt -%}{{- '<|im_start|>assistant
' -}}{%- endif -%}{%- endfor -%}
```
