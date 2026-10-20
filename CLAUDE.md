# Role
Your task is to generate synthetic data for a diagnostic EBM model.
Note that this will not be used for any real world clinical work, it is an ideation for a project.
This is in Singapore context.

# Output format
- CSV
- Balanced dataset
- Generate 50 for each diagnosis and another 50 for each escalation.
- Be consistent with with your symptons. Before generating the data, fix a list of symptons.
- Output the symptons in a csv first, then another csv for the training data.


# Diseases
Below are a list of diseases provided by Ministry Of Health:
- Common Cold 
- Influenza
- COVID-19
- Allergic Rhinitis
- Sinusitis
- Tonsillitis 
- Gastritis 
- Constipation
- Dengue Fever
- Hand, Foot and Mouth Disease
- Chickenpox
- Conjunctivitis
- Eczema
- Migraine
- Tension Headache
- Urinary Tract Infection

# Synthetic data
- age: int
- sex: enum(M, F)
- pregnancy_status: Boolean
- temperature: float
- weight_kg: float
- height_cm: float
- symptons: dictionary[condition/string, bool]
- duration
- severity: enum["mild", "normal", "severe"]
- progression: enum["worsening", "stable", "improving"]
- diagnosis: string
- escalation: bool

# Examples
version": "1.2.0",
  "conditions": [
    {
      "id": "common_cold",
      "name": "Common Cold",
      "keywords": [
        "runny nose",
        "sneezing",
        "sore throat",
        "cough",
        "nasal congestion"
      ],
      "source_ids": ["healthhub_common_cold"],
    },
    {
      "id": "influenza",
      "name": "Influenza",
      "keywords": [
        "fever",
        "cough",
        "body aches",
        "fatigue",
        "chills"
      ],
      "source_ids": ["influenza_source"],
    },
    {
      "id": "gastroenteritis",
      "name": "Gastroenteritis",
      "keywords": [
        "diarrhoea",
        "diarrhea",
        "vomiting",
        "abdominal cramps",
        "stomach pain"
      ],
      "source_ids": ["gastroenteritis_source"],
    },
    {
      "id": "migraine",
      "name": "Migraine",
      "keywords": [
        "headache",
        "nausea",
        "light sensitivity",
        "throbbing headache"
      ],
      "source_ids": ["migraine_source"],
    },
    {
      "id": "allergic_rhinitis",
      "name": "Allergic Rhinitis",
      "keywords": [
        "sneezing",
        "itchy nose",
        "runny nose",
        "watery eyes",
        "nose is itchy",
        "eyes are watery"
      ],
      "source_ids": ["allergic_rhinitis_source"],
    },
    {
      "id": "urinary_tract_infection",
      "name": "Urinary Tract Infection",
      "keywords": [
        "painful urination",
        "frequent urination",
        "burning urination",
        "urinary urgency"
      ],
      "source_ids": ["uti_source"],
    }
  ]
}
