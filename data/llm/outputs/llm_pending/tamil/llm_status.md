# LLM Comparison Status

- model_name: `llm_pending`
- language: `tamil`
- status: `waiting_for_parsed_conllu`

No parsed CoNLL-U input was provided, so this stage generated only the prompt template.

Next step:

1. Generate sentences with an LLM using `data/llm/prompts/llm_generation_prompts.csv`
2. Parse the generated text into CoNLL-U
3. Re-run `run_llm_comparison.py` with `--conllu-path`
