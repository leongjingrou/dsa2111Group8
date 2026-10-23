# Synthetic scenario generator

`generate.py` creates fictional scenarios for the DSA2111 project prototype.
It does not use real patient records and is not for clinical use.

Run it from the repository root:

```bash
python data/generate.py
```

The generator is also callable as `data.generate.generate(output_dir)`. The
scenario-model commands expose this through `--generate-scenarios`, so model
training can regenerate and validate its input rather than relying implicitly on
the checked-in CSV.

Generated files are written to `data/generated/`:

- `synthetic_scenarios.csv` is the complete pool of 1,700 structured scenarios across 17 fictional diagnosis labels, before any train/validation/test split. Six labels match the current knowledge-base scope; the rest are marked `out_of_scope`. Each diagnosis has 50 rows with `escalation=false` and 50 with `escalation=true`.
- `symptoms.csv` contains the symptom vocabulary and the generator's illustrative associations.

The scenario CSV keeps structured values such as age, sex, duration, severity, progression,
the assessed symptom answers, diagnosis, escalation label, scope status,
synthetic marker, and dataset version. It also adds `vignette_text`, a readable
summary rendered from the same row. A separate dialogue is not included until
the project implements conversational question and answer behavior.

The random seed is fixed at `42` and each diagnosis/escalation combination has
50 rows. Generation probabilities and escalation labels are invented for
software demonstration. They are not medical advice, clinical ground truth,
or evidence of model safety or performance. The generated file is not directly
compatible with `diagnostic_model/training.py`: a separate feature-selection
and encoding step is needed before model training.
