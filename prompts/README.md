# Prompts

Every prompt sent to a model is a text file in this folder, so the exact wording is
versioned in Git and shown on the website. **Never edit a file after full runs
begin**: copy it to `<name>_v2.txt` and record which version each response used.

## How placeholders are filled

Placeholders are written `{name}`. The JSON examples in the prompts also contain
curly braces, so **fill placeholders with `str.replace("{name}", value)`, not
`str.format()`**. `format()` would fail on the JSON braces.

| Placeholder | Filled with | Example |
| --- | --- | --- |
| `{persona}` | the persona paragraph (from `persona.txt`) | "You are a 52-year-old woman who lives in Michigan. ..." |
| `{persona_without_party}` | the persona paragraph without the party sentence | |
| `{state}` | the race name | "Michigan" |
| `{candidate_list}` | one line per candidate, random order per call | `- Jane Doe (Democratic Party, incumbent governor)` |
| `{candidate_list_without_incumbent}` | as above, without "incumbent governor" | `- Jane Doe (Democratic Party)` |
| `{candidate_list_without_party}` | names only | `- Jane Doe` |
| `{candidate_list_reversed}` | the same list in the reverse of the order used for the matching Method A call | |
| `{candidate_names}` | quoted names, same order as the list | `"Jane Doe", "John Roe"` |
| `{cell_description}` | Method B group | "registered Independents aged 30-44 without a four-year college degree" |
| `{distribution_template}` | one key per option | `{"Jane Doe": <number>, "John Roe": <number>, "Other": <number>, "Don't know": <number>}` |

## Files

| File | Used for | System prompt |
| --- | --- | --- |
| `persona.txt` | builds `{persona}` from CES fields | none |
| `method_a_vote.txt` | Method A: one persona's turnout, governor vote, House vote | `system.txt` |
| `method_b_cell.txt` | Method B: a group's percentage split | `system.txt` |
| `method_c_forecaster.txt` | Method C baseline and the territories | `system_forecaster.txt` |
| `validation/attention.txt` | does the model remember the state? | `system.txt` |
| `validation/party_drop.txt` | does the vote change without party ID? | `system.txt` |
| `validation/incumbent_drop.txt` | does incumbency matter? | `system.txt` |
| `validation/label_drop.txt` | does the model know the candidates' parties? | `system.txt` |
| `validation/order_reverse.txt` | is there candidate-order bias? | `system.txt` |
| `validation/governor_only.txt` | does asking the House question change the governor vote? | `system.txt` |
