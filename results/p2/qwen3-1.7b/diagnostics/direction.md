# Direction diagnostic · qwen3-1.7b · sft

First 100 FLORES devtest sentences, greedy decoding, same generation code as the evaluation. *Same as stored* compares with the outputs saved by `p2_translate.py` (base model / merged adapter); a few differences are expected from different batching (padding) in half precision.

| Condition | In target language % | Exact copy of source % | chrF | Languages | Same as stored % |
|---|---|---|---|---|---|
| de→en, adapter off | 100 | 0 | 63.9 | en 100 | 98 (base) |
| de→en, adapter ×0.25 | 100 | 0 | 64.0 | en 100 | – |
| de→en, adapter ×0.5 | 100 | 0 | 62.8 | en 100 | – |
| de→en, adapter ×0.75 | 100 | 0 | 62.6 | en 100 | – |
| de→en, adapter ×1 (SFT) | 100 | 0 | 61.6 | en 100 | 98 (sft) |
| de→en, SFT, instruction after the text | 100 | 0 | 61.6 | en 100 | – |
| fr→en, adapter off | 100 | 0 | 64.9 | en 100 | – |
| fr→en, SFT | 100 | 0 | 63.4 | en 100 | – |
| de→zh, adapter off | 100 | 0 | 30.5 | zh 100 | – |
| de→zh, SFT | 95 | 0 | 28.0 | zh 95, en 5 | – |
| en→zh, SFT (control) | 9 | 32 | 6.4 | en 91, zh 9 | – |
| fr→de, SFT (control) | 100 | 0 | 47.5 | de 100 | – |

Examples (first 3 sentences):

- *de→en, adapter off*: "We now have 4-month-old mice that had diabetes and no longer have any left," he added.
- *de→en, adapter off*: Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia, and chair of the Department of Clinical and Scientific Research of the Canad
- *de→en, adapter off*: As some other experts showed, he was skeptical about whether diabetes could be cured, and pointed out that the findings for people who already had type 1 diabet
- *de→en, adapter ×0.5*: “We now have 4-month-old mice that have diabetes and no longer have them,” he added.
- *de→en, adapter ×0.5*: Dr. Ehud Ur, a professor of medicine at Dalhousie University in Halifax, Nova Scotia, and chair of the Canadian Diabetes Association’s Department of Clinical an
- *de→en, adapter ×0.5*: As some other experts showed, he was skeptical about the possibility of curing diabetes, and pointed out that the findings for people who already had type 1 dia
- *de→en, adapter ×1 (SFT)*: “We now have 4-month-old mice that have diabetes, and we don’t have any more,” he added.
- *de→en, adapter ×1 (SFT)*: Dr. Ehud Ur, a professor of medicine at Dalhousie University in Halifax, Nova Scotia, and chair of the Canadian Diabetes Association’s Department of Clinical an
- *de→en, adapter ×1 (SFT)*: As some other experts pointed out, he was skeptical about the possibility of curing diabetes, and noted that the findings for people who already had type 1 diab
- *fr→en, SFT*: “We now have 4-month-old mice that are not diabetic, even though they were diabetic before,” he added.
- *fr→en, SFT*: Dr. Ehud Ur, a professor of medicine at Dalhousie University in Halifax, Nova Scotia, and president of the Canadian Diabetes Association's Clinical and Scientif
- *fr→en, SFT*: Like other experts, he is skeptical about the possibility of curing diabetes, noting that these results do not apply to people who already have type 1 diabetes.
- *de→zh, SFT*: “我们现在已经有了4个月大的老鼠，它们有糖尿病，但现在没有了。”
- *de→zh, SFT*: Dr. Ehud Ur, 医学教授, 位于新斯科舍省哈利法克斯的多伦多大学, 以及加拿大糖尿病协会的临床与研究部门负责人, 认为研究工作还处于起步阶段。
- *de→zh, SFT*: 正如一些专家所指出的那样，他表现出对治愈糖尿病的怀疑态度，并指出，对于那些已经患有1型糖尿病的人，这些研究结果是没有意义的。
