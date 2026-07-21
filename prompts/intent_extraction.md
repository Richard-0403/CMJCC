# Intent Extraction Prompt

You extract structured job-search preferences from a single candidate utterance.

Return ONLY a JSON object conforming to the extraction schema. Do not add prose,
explanations, or chain-of-thought. For each preference provide:

- `field_name`: one of target_roles, skills_have, preferred_locations,
  work_modes, salary_min, salary_currency, experience_level, years_experience,
  excluded_roles, excluded_locations.
- `normalized_value`: the normalised value.
- `raw_text`: the exact span from the utterance.
- `confidence`: 0..1.
- `proposed_strength`: hard | soft | unknown | not_applicable. Use `hard` for
  "must / only / at least / above / minimum / cannot"; `soft` for
  "prefer / ideally / flexible / also fine"; `unknown` for "maybe / not sure".
- `polarity`: positive | negative (negative for "don't want / exclude").
- `temporal_scope`: current_search | session | long_term | unknown.

Never invent facts that are not present in the utterance.

Utterance:
{utterance}
