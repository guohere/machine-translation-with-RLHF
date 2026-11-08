# Direction diagnostic · qwen3-1.7b · sft

First 100 FLORES devtest sentences, greedy decoding, same generation code as the evaluation. *Same as stored* compares with the outputs saved by `p2_translate.py` (base model / merged adapter); a few differences are expected from different batching (padding) in half precision.

| Condition | In target language % | Exact copy of source % | chrF | Languages | Same as stored % |
|---|---|---|---|---|---|
| de→en, adapter off | 100 | 0 | 63.9 | en 100 | 98 (base) |
| de→en, adapter ×0.25 | 98 | 0 | 63.8 | en 98, de 2 | – |
| de→en, adapter ×0.5 | 65 | 14 | 49.3 | en 65, de 35 | – |
| de→en, adapter ×0.75 | 13 | 35 | 31.3 | de 87, en 13 | – |
| de→en, adapter ×1 (SFT) | 2 | 32 | 27.9 | de 98, en 2 | 98 (sft) |
| de→en, SFT, instruction after the text | 37 | 20 | 38.4 | de 63, en 37 | – |
| fr→en, adapter off | 100 | 0 | 64.9 | en 100 | – |
| fr→en, SFT | 20 | 36 | 36.6 | fr 80, en 20 | – |
| de→zh, adapter off | 100 | 0 | 30.5 | zh 100 | – |
| de→zh, SFT | 5 | 32 | 4.6 | de 95, zh 5 | – |
| en→zh, SFT (control) | 95 | 1 | 31.6 | zh 95, en 3, de 1, ja 1 | – |
| fr→de, SFT (control) | 100 | 0 | 48.4 | de 100 | – |

Examples (first 3 sentences):

- *de→en, adapter off*: "We now have 4-month-old mice that had diabetes and no longer have any left," he added.
- *de→en, adapter off*: Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia, and chair of the Department of Clinical and Scientific Research of the Canad
- *de→en, adapter off*: As some other experts showed, he was skeptical about whether diabetes could be cured, and pointed out that the findings for people who already had type 1 diabet
- *de→en, adapter ×0.5*: „Wir haben jetzt 4 Monate alte Mäuse, die Diabetes hatten und jetzt keinen mehr haben“, fügte er hinzu.
- *de→en, adapter ×0.5*: Dr. Ehud Ur, Professor of Medicine at Dalhousie University in Halifax, Nova Scotia, and President of the Canadian Diabetes Association's Department of Clinical 
- *de→en, adapter ×0.5*: Like some other experts, he was skeptical about the possibility of curing diabetes and pointed out that the findings for people who already had type 1 diabetes 
- *de→en, adapter ×1 (SFT)*: „Wir haben jetzt 4 Monate alte Mäuse, die Diabetes hatten und jetzt keinen mehr haben“, fügte er hinzu.
- *de→en, adapter ×1 (SFT)*: Dr. Ehud Ur, Professor für Medizin an der Dalhousie University in Halifax, Nova Scotia, und Vorsitzender der Abteilung für Klinik und Wissenschaft des Kanadisch
- *de→en, adapter ×1 (SFT)*: Wie einige andere Experten zeigte er sich skeptisch, ob es möglich sei, Diabetes zu heilen, und wies darauf hin, dass die Befunde für Menschen, die bereits unte
- *fr→en, SFT*: « Nous avons maintenant des souris de 4 mois qui ne sont pas diabétiques alors qu'elles l'étaient auparavant », a-t-il ajouté.
- *fr→en, SFT*: Dr. Ehud Ur, médecin à l'Université Dalhousie de Halifax (Nouvelle-Écosse) et président de la division clinique et scientifique de l'Association canadienne du d
- *fr→en, SFT*: Ainsi qu'ont fait d'autres experts, il est sceptique quant à la possibilité de guérir le diabète, en soulignant que ces résultats ne s'appliquent pas aux person
- *de→zh, SFT*: „Wir haben jetzt 4 Monate alte Mäuse, die Diabetes hatten und jetzt keinen mehr haben“, fügte er hinzu.
- *de→zh, SFT*: Dr. Ehud Ur, Professor für Medizin an der Dalhousie University in Halifax, Nova Scotia, und Vorsitzender der Abteilung für Klinik und Wissenschaft des Kanadisch
- *de→zh, SFT*: Wie einige andere Experten zeigte er sich skeptisch, ob es möglich sei, Diabetes zu heilen, und wies darauf hin, dass die Befunde für Menschen, die bereits unte
