# Grounding plan: open medical data and Southeast Asian context

Reviewed 25 October 2026. This is a provenance and research-use plan, not a
claim that any source validates clinical deployment.

## Recommended source lanes

| Source | Region | What it contributes | Licence / decision |
|---|---|---|---|
| [DDXPlus](https://github.com/mila-iqia/ddxplus) | Global synthetic | Structured symptoms, antecedents, pathology, and probability-weighted differential diagnosis; official train/validation/test splits | CC BY 4.0; primary diagnosis research baseline |
| [CARE-Bench](https://github.com/ningkko/CARE-bench) | Source-grounded, not SEA-specific | Patient-disclosure prefixes labelled as ask, self-care, non-urgent care, or urgent care | CC BY-NC 4.0 dataset; evaluation only, never train on the held-out split; repository code licence unspecified |
| [OpenMed MedDialog](https://huggingface.co/datasets/OpenMed/MedDialog) | Global English online Q&A | Patient wording and clinician-response style | Apache-2.0 as stated on the card; language audit only because responses vary in quality and there are no dependable diagnosis labels |
| [Singapore MOH: Getting medical help](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/) | Singapore | Local emergency warning signs and care pathway | Encoded as an attributed, review-dated safety override; not model training data |
| [Singapore MOH condition catalogue](https://data.gov.sg/datasets/d_01d45cd7b2113dc0c433bcd5218b67b8/view) | Singapore | Audit of locally published condition coverage | Singapore Open Data Licence; catalogue only (42 names, no patient records) |
| [ViMed-SFT](https://huggingface.co/datasets/quannguyen204/medical_sft_crawl_vi_10k_v1) | Vietnam | 10,085 Vietnamese medical QA conversations | Apache-2.0 on the card; pilot language evaluation only until crawl provenance is audited |
| [IndoHealth-NLP NCD sample](https://huggingface.co/datasets/IndoHealth-NLP/NCD_Instruct-Tuning_Medical_QA_Indonesian) | Indonesia | 200 public Indonesian NCD education/triage examples | CC BY-NC 4.0; non-commercial language evaluation, not diagnosis training |
| [SEADialogues](https://huggingface.co/datasets/SEACrowd/SEADialogues) | Six Southeast Asian countries, eight languages | Cultural entities, personas, register, and multi-turn language robustness | CC BY-SA 4.0; non-medical auxiliary only |
| [ViMQ](https://github.com/tadeephuy/ViMQ) | Vietnam | 9,000 patient questions with intent and medical-entity labels | Hold until repository reuse terms are confirmed |
| [Meddies Consultant](https://huggingface.co/datasets/HoangHa/medical-data) | Vietnam | Large Vietnamese-first multi-turn synthetic consultations | Hold: card reports an “other” licence and generation/provenance needs audit |

DDXPlus should be cited as: Fansi Tchango, A., Goel, R., Wen, Z., Martel, J.,
and Ghosn, J. (2022), “DDXPlus: A New Dataset for Automatic Medical
Diagnosis,” *NeurIPS 2022 Datasets and Benchmarks Track*,
[arXiv:2205.09148](https://arxiv.org/abs/2205.09148). The corresponding data and
documentation are the [Mila-IQIA DDXPlus repository](https://github.com/mila-iqia/ddxplus).
The server pipeline fixes the English data release to Figshare version 2,
[doi:10.6084/m9.figshare.22687585.v2](https://doi.org/10.6084/m9.figshare.22687585.v2),
and hashes every extracted source file.

In this project, “DDXPlus-grounded” means that age, sex, evidence values,
simulated pathology, differential distribution, and official split come from a
DDXPlus row. The doctor/patient wording is a deterministic reconstruction from
the dataset's English question metadata and is explicitly marked
`reconstruction: true`; it is not presented as an observed clinical exchange.

## What “Southeast Asian grounding” means here

There is no open, clinically validated Singaporean doctor–patient diagnostic
dialogue corpus in the sources reviewed. Regional grounding therefore must not
be simulated by relabelling a global corpus. The proposed evaluation has four
independent axes:

1. **Clinical content:** DDXPlus for diagnosis and differential distributions.
2. **Question sufficiency:** CARE-Bench for whether the system should ask,
   recommend routine care, or escalate.
3. **Jurisdictional safety:** an optional Singapore MOH profile can override
   learned stopping confidence; it is disabled by default, must be selected
   explicitly, and the source plus profile decision appear in every affected
   log alongside the untouched model decision.
4. **Language and culture:** English, Vietnamese, Indonesian, Malay, Tamil,
   Tagalog and Thai paraphrases are scored separately for extraction stability,
   question count, urgent recall, calibration and abstention. SEADialogues may
   inform register and cultural robustness but never medical labels.

Results must be reported per country/language and never collapsed into a single
“SEA score.” Local clinician review is still required before adding a language
or condition to the supported set.

## Seeded generation contract

The V4 generator uses DDXPlus as the medical seed and public ViMed-SFT and
IndoHealth preview rows as Southeast Asian style seeds. It assigns every
regional record to exactly one train/validation/test partition by a stable hash
and pairs only within that partition. A generated record carries both source
IDs, URLs, licences, record IDs, a hash/excerpt of the style seed, and the
generation seed.

The regional text is deliberately not inserted as clinical evidence. It cannot
change the DDXPlus pathology, differential, or evidence tokens. This produces a
traceable augmentation/evaluation input while preventing an unverified QA
answer or a non-medical cultural corpus from becoming a diagnosis label. It is
still synthetic and does not establish Southeast Asian clinical validity.

## Leakage and licensing controls

- Keep each source's official split and source ID; deduplicate by normalized
  patient text before combining sources.
- Do not train on CARE-Bench public-test or controlled-test examples.
- Do not infer a diagnosis label from MedDialog or regional QA assistant text.
- Keep non-commercial and share-alike sources in separate output trees.
- Store raw downloads outside Git; commit only adapters, hashes, aggregate
  audits and small licence-compatible fixtures.
- Log whether a stop came from the learned controller, the five-answer cap, or
  the Singapore safety override.
