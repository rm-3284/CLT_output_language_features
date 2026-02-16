Execute the python files in the following order to replicate the experiments.

# CLT Output Language Features

This directory finds language-controlling latents in three ways:

1. **Annotation-based**: Using Neuronpedia feature descriptions
2. **Value-based**: Measuring differential activation across languages
3. **Frequency-based**: Counting activation frequency across languages and examples

## Pipeline: Generating Intervention Results

To produce comprehensive intervention results in `data/interventions/`, execute the following files in order:

```
python flores_feature_extraction.py
python language_specific_features.py
python multilingual_llm_features.py
python amplification_values.py
python interventions_to_json.py
```

This generates:

- `data/interventions/{prompt_lang}/{adj_lang}/interventions_and_results_description.json` (annotation-based)
- `data/interventions/{prompt_lang}/{adj_lang}/interventions_and_results_value.json` (value-based)
- `data/interventions/{prompt_lang}/{adj_lang}/interventions_and_results_frequency.json` (frequency-based)

## Legacy Pipeline

Original execution order for other experiments:

```
python confirming_behavior.py
python feature_extraction.py
python supernode_threshold.py
python feature_values_for_generic_sentences.py \approx python amplification_values.py
python feature_classify_to_supernode.py
python chinese_ablation_strength.py
python ablation_amplification_intervention.py
```
