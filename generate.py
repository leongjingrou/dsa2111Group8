"""Synthetic data generator for a diagnostic EBM model (Singapore context).

NOT for clinical use. Produces:
  - symptoms.csv : fixed symptom vocabulary
  - training_data.csv : 16 diagnoses x (50 non-escalated + 50 escalated) = 1600 rows

Symptoms are stored as a JSON object of {symptom_id: bool} containing only the
symptoms a doctor would plausibly have asked about for that patient: everything
the patient reported, most of the presenting disease's typical symptoms, some
symptoms used to rule out differential diagnoses, and screening red flags.
"""

import csv
import json
import random

SEED = 42
N_PER_CLASS = 50  # per diagnosis, per escalation value

# ---------------------------------------------------------------------------
# Fixed symptom vocabulary: (id, name, category, is_red_flag)
# ---------------------------------------------------------------------------
SYMPTOMS = [
    # Respiratory / ENT
    ("runny_nose", "Runny nose", "respiratory", False),
    ("nasal_congestion", "Nasal congestion", "respiratory", False),
    ("sneezing", "Sneezing", "respiratory", False),
    ("itchy_nose", "Itchy nose", "respiratory", False),
    ("sore_throat", "Sore throat", "respiratory", False),
    ("cough", "Cough", "respiratory", False),
    ("thick_nasal_discharge", "Thick yellow/green nasal discharge", "respiratory", False),
    ("facial_pain_pressure", "Facial pain or pressure", "respiratory", False),
    ("loss_of_smell_taste", "Loss of smell or taste", "respiratory", False),
    ("difficulty_swallowing", "Difficulty or pain swallowing", "respiratory", False),
    ("swollen_tonsils", "Red, swollen tonsils", "respiratory", False),
    ("swollen_neck_glands", "Swollen neck glands", "respiratory", False),
    ("wheezing", "Wheezing", "respiratory", True),
    ("shortness_of_breath", "Shortness of breath", "respiratory", True),
    ("chest_pain", "Chest pain", "respiratory", True),
    ("difficulty_opening_mouth", "Difficulty opening mouth / drooling", "respiratory", True),
    # Systemic
    ("fever", "Fever", "systemic", False),
    ("chills", "Chills", "systemic", False),
    ("body_aches", "Body aches", "systemic", False),
    ("fatigue", "Fatigue", "systemic", False),
    ("loss_of_appetite", "Loss of appetite", "systemic", False),
    ("joint_muscle_pain", "Severe joint or muscle pain", "systemic", False),
    ("confusion_drowsiness", "Confusion or unusual drowsiness", "systemic", True),
    ("dehydration_signs", "Signs of dehydration (reduced urine, dry mouth)", "systemic", True),
    ("unintentional_weight_loss", "Unintentional weight loss", "systemic", True),
    # Gastrointestinal
    ("upper_abdominal_pain", "Upper abdominal / epigastric pain", "gastrointestinal", False),
    ("lower_abdominal_pain", "Lower abdominal pain", "gastrointestinal", False),
    ("bloating", "Bloating", "gastrointestinal", False),
    ("heartburn", "Heartburn / indigestion", "gastrointestinal", False),
    ("nausea", "Nausea", "gastrointestinal", False),
    ("vomiting", "Vomiting", "gastrointestinal", False),
    ("infrequent_bowel_movements", "Fewer than 3 bowel movements a week", "gastrointestinal", False),
    ("hard_stools", "Hard or lumpy stools", "gastrointestinal", False),
    ("straining", "Straining during bowel movements", "gastrointestinal", False),
    ("persistent_vomiting", "Persistent vomiting", "gastrointestinal", True),
    ("vomiting_blood", "Vomiting blood", "gastrointestinal", True),
    ("black_stools", "Black, tarry stools", "gastrointestinal", True),
    ("blood_in_stool", "Blood in stool", "gastrointestinal", True),
    ("severe_abdominal_pain", "Severe abdominal pain", "gastrointestinal", True),
    # Skin
    ("rash", "Skin rash", "skin", False),
    ("itchy_blisters", "Itchy fluid-filled blisters", "skin", False),
    ("mouth_ulcers", "Mouth ulcers", "skin", False),
    ("blisters_hands_feet", "Spots/blisters on palms and soles", "skin", False),
    ("dry_itchy_skin", "Dry, itchy skin", "skin", False),
    ("red_inflamed_patches", "Red, inflamed skin patches", "skin", False),
    ("skin_infection_signs", "Weeping, crusting or pus on skin", "skin", True),
    ("bleeding_gums_nose", "Bleeding gums or nose", "skin", True),
    # Eye
    ("red_eyes", "Red eyes", "eye", False),
    ("watery_eyes", "Watery eyes", "eye", False),
    ("itchy_eyes", "Itchy eyes", "eye", False),
    ("eye_discharge", "Eye discharge", "eye", False),
    ("pain_behind_eyes", "Pain behind the eyes", "eye", False),
    ("eye_pain", "Eye pain", "eye", True),
    ("blurred_vision", "Blurred vision", "eye", True),
    ("periorbital_swelling", "Swelling around the eye", "eye", True),
    # Neurological
    ("headache", "Headache", "neurological", False),
    ("throbbing_headache", "Throbbing, one-sided headache", "neurological", False),
    ("band_like_headache", "Tight, band-like headache", "neurological", False),
    ("light_sensitivity", "Sensitivity to light", "neurological", False),
    ("sound_sensitivity", "Sensitivity to sound", "neurological", False),
    ("visual_aura", "Visual aura", "neurological", False),
    ("neck_shoulder_tension", "Neck or shoulder muscle tension", "neurological", False),
    ("neck_stiffness", "Neck stiffness", "neurological", True),
    ("sudden_severe_headache", "Sudden, worst-ever headache", "neurological", True),
    ("weakness_numbness", "Weakness or numbness", "neurological", True),
    # Urinary
    ("painful_urination", "Painful / burning urination", "urinary", False),
    ("frequent_urination", "Frequent urination", "urinary", False),
    ("urinary_urgency", "Urinary urgency", "urinary", False),
    ("cloudy_urine", "Cloudy or foul-smelling urine", "urinary", False),
    ("blood_in_urine", "Blood in urine", "urinary", False),
    ("flank_pain", "Flank / back pain", "urinary", True),
]
SYMPTOM_IDS = [s[0] for s in SYMPTOMS]
BACKGROUND_P = 0.03  # chance any non-red-flag symptom appears as noise
FEVER_THRESHOLD = 37.5

# Probabilities that a doctor asks about a symptom the patient did not report
ASK_CORE_P = 0.8          # typical symptoms of the presenting disease
ASK_DIFFERENTIAL_P = 0.4  # typical symptoms of a confusable disease
ASK_RED_FLAG_P = 0.5      # red-flag screening questions
ASK_FEVER_P = 0.9

# ---------------------------------------------------------------------------
# Disease profiles
#   core:      symptom -> probability present
#   red_flags: symptom -> probability present when escalated
#   temp:      (normal_low, normal_high, escalated_low, escalated_high) in °C
#   dur:       (normal_low, normal_high, escalated_low, escalated_high) in days
#   age:       (min, max); f_ratio: probability of female
#   risk_age:  predicate on age that raises escalation plausibility
# ---------------------------------------------------------------------------
DISEASES = {
    "Common Cold": dict(
        core={"runny_nose": .9, "nasal_congestion": .85, "sneezing": .8, "sore_throat": .7,
              "cough": .65, "fatigue": .4, "headache": .3, "body_aches": .2},
        red_flags={"shortness_of_breath": .6, "wheezing": .4, "chest_pain": .2,
                   "dehydration_signs": .3, "confusion_drowsiness": .1},
        temp=(36.5, 37.8, 38.3, 39.5), dur=(2, 7, 10, 18),
        age=(1, 80), f_ratio=.5, risk_age=lambda a: a < 2 or a > 70),
    "Influenza": dict(
        core={"cough": .85, "body_aches": .85, "fatigue": .9, "chills": .75, "headache": .65,
              "sore_throat": .5, "runny_nose": .4, "loss_of_appetite": .5},
        red_flags={"shortness_of_breath": .7, "chest_pain": .4, "confusion_drowsiness": .35,
                   "persistent_vomiting": .25, "dehydration_signs": .35},
        temp=(38.0, 39.2, 39.3, 40.5), dur=(2, 6, 5, 12),
        age=(1, 88), f_ratio=.5, risk_age=lambda a: a < 5 or a >= 65),
    "COVID-19": dict(
        core={"cough": .8, "sore_throat": .65, "fatigue": .75, "loss_of_smell_taste": .4,
              "runny_nose": .45, "headache": .5, "body_aches": .5, "chills": .35},
        red_flags={"shortness_of_breath": .8, "chest_pain": .45, "confusion_drowsiness": .3,
                   "dehydration_signs": .2},
        temp=(37.3, 38.8, 38.5, 40.2), dur=(2, 7, 6, 14),
        age=(3, 90), f_ratio=.5, risk_age=lambda a: a >= 65),
    "Allergic Rhinitis": dict(
        core={"sneezing": .9, "itchy_nose": .85, "runny_nose": .85, "nasal_congestion": .7,
              "watery_eyes": .65, "itchy_eyes": .6, "fatigue": .2},
        red_flags={"wheezing": .7, "shortness_of_breath": .55, "chest_pain": .15},
        temp=(36.3, 37.3, 36.4, 37.5), dur=(7, 60, 30, 120),
        age=(4, 65), f_ratio=.5, risk_age=lambda a: False),
    "Sinusitis": dict(
        core={"nasal_congestion": .9, "thick_nasal_discharge": .85, "facial_pain_pressure": .85,
              "headache": .6, "loss_of_smell_taste": .35, "cough": .4, "fatigue": .35},
        red_flags={"periorbital_swelling": .6, "blurred_vision": .35, "confusion_drowsiness": .25,
                   "neck_stiffness": .2, "sudden_severe_headache": .25},
        temp=(36.6, 38.0, 38.3, 39.6), dur=(5, 12, 12, 35),
        age=(10, 75), f_ratio=.55, risk_age=lambda a: a > 70),
    "Tonsillitis": dict(
        core={"sore_throat": .95, "difficulty_swallowing": .85, "swollen_tonsils": .9,
              "swollen_neck_glands": .7, "headache": .35, "loss_of_appetite": .45, "fatigue": .4},
        red_flags={"difficulty_opening_mouth": .7, "shortness_of_breath": .35,
                   "dehydration_signs": .5},
        temp=(37.5, 38.8, 39.0, 40.3), dur=(2, 6, 4, 10),
        age=(3, 35), f_ratio=.5, risk_age=lambda a: False),
    "Gastritis": dict(
        core={"upper_abdominal_pain": .9, "bloating": .65, "nausea": .65, "heartburn": .6,
              "loss_of_appetite": .45, "vomiting": .25},
        red_flags={"black_stools": .55, "vomiting_blood": .45, "persistent_vomiting": .35,
                   "unintentional_weight_loss": .35, "severe_abdominal_pain": .3},
        temp=(36.3, 37.3, 36.4, 37.8), dur=(2, 14, 10, 45),
        age=(18, 80), f_ratio=.5, risk_age=lambda a: a > 55),
    "Constipation": dict(
        core={"infrequent_bowel_movements": .9, "hard_stools": .85, "straining": .8,
              "bloating": .55, "lower_abdominal_pain": .5, "loss_of_appetite": .15},
        red_flags={"blood_in_stool": .5, "persistent_vomiting": .3, "severe_abdominal_pain": .5,
                   "unintentional_weight_loss": .35},
        temp=(36.2, 37.2, 36.4, 38.0), dur=(3, 14, 14, 60),
        age=(2, 88), f_ratio=.6, risk_age=lambda a: a > 60),
    "Dengue Fever": dict(
        core={"headache": .85, "pain_behind_eyes": .75, "joint_muscle_pain": .85, "rash": .5,
              "nausea": .55, "fatigue": .8, "chills": .5, "loss_of_appetite": .6},
        red_flags={"severe_abdominal_pain": .6, "persistent_vomiting": .55,
                   "bleeding_gums_nose": .55, "confusion_drowsiness": .3,
                   "dehydration_signs": .35},
        temp=(38.2, 39.5, 38.0, 40.5), dur=(2, 5, 3, 8),
        age=(5, 80), f_ratio=.5, risk_age=lambda a: a > 65),
    "Hand, Foot and Mouth Disease": dict(
        core={"mouth_ulcers": .9, "blisters_hands_feet": .9, "rash": .55, "sore_throat": .5,
              "loss_of_appetite": .7, "fatigue": .4},
        red_flags={"confusion_drowsiness": .55, "persistent_vomiting": .45,
                   "neck_stiffness": .3, "dehydration_signs": .6, "weakness_numbness": .15},
        temp=(37.5, 38.6, 38.8, 40.0), dur=(1, 6, 3, 9),
        age=(1, 10), f_ratio=.5, risk_age=lambda a: a < 3),
    "Chickenpox": dict(
        core={"itchy_blisters": .95, "rash": .85, "fatigue": .55, "loss_of_appetite": .45,
              "headache": .35},
        red_flags={"shortness_of_breath": .45, "skin_infection_signs": .5,
                   "confusion_drowsiness": .3, "neck_stiffness": .15, "chest_pain": .15},
        temp=(37.4, 38.6, 38.8, 40.0), dur=(1, 7, 5, 12),
        age=(1, 45), f_ratio=.5, risk_age=lambda a: a < 1 or a >= 13),
    "Conjunctivitis": dict(
        core={"red_eyes": .95, "eye_discharge": .75, "watery_eyes": .7, "itchy_eyes": .6},
        red_flags={"eye_pain": .75, "blurred_vision": .6, "light_sensitivity": .55,
                   "periorbital_swelling": .3},
        temp=(36.3, 37.4, 36.5, 38.0), dur=(1, 7, 5, 14),
        age=(1, 75), f_ratio=.5, risk_age=lambda a: a < 1),
    "Eczema": dict(
        core={"dry_itchy_skin": .95, "red_inflamed_patches": .85, "rash": .6},
        red_flags={"skin_infection_signs": .85, "chills": .3},
        temp=(36.2, 37.2, 37.5, 39.0), dur=(7, 60, 14, 90),
        age=(1, 45), f_ratio=.5, risk_age=lambda a: a < 2),
    "Migraine": dict(
        core={"throbbing_headache": .9, "headache": .95, "light_sensitivity": .8,
              "sound_sensitivity": .65, "nausea": .65, "visual_aura": .3, "vomiting": .25},
        red_flags={"sudden_severe_headache": .6, "weakness_numbness": .5,
                   "confusion_drowsiness": .35, "neck_stiffness": .3, "blurred_vision": .3},
        temp=(36.3, 37.3, 36.4, 38.2), dur=(1, 3, 1, 7),
        age=(12, 60), f_ratio=.72, risk_age=lambda a: a > 50),
    "Tension Headache": dict(
        core={"band_like_headache": .9, "headache": .95, "neck_shoulder_tension": .8,
              "fatigue": .45, "light_sensitivity": .15},
        red_flags={"sudden_severe_headache": .55, "weakness_numbness": .45,
                   "confusion_drowsiness": .35, "neck_stiffness": .35, "blurred_vision": .3},
        temp=(36.2, 37.2, 36.4, 38.2), dur=(1, 10, 7, 30),
        age=(16, 70), f_ratio=.6, risk_age=lambda a: a > 50),
    "Urinary Tract Infection": dict(
        core={"painful_urination": .9, "frequent_urination": .85, "urinary_urgency": .8,
              "cloudy_urine": .55, "lower_abdominal_pain": .55, "blood_in_urine": .2},
        red_flags={"flank_pain": .75, "chills": .6, "vomiting": .45,
                   "confusion_drowsiness": .15},
        temp=(36.5, 37.8, 38.3, 40.0), dur=(1, 5, 3, 10),
        age=(16, 85), f_ratio=.85, risk_age=lambda a: a > 70),
}

# Differentials: diseases sharing at least one typical, non-generic symptom
GENERIC_SYMPTOMS = {"fatigue", "loss_of_appetite", "headache", "body_aches"}
DIFFERENTIALS = {
    d: [o for o, op in DISEASES.items()
        if o != d and (set(p["core"]) & set(op["core"])) - GENERIC_SYMPTOMS]
    for d, p in DISEASES.items()
}

SEVERITIES = ["mild", "normal", "severe"]
PROGRESSIONS = ["worsening", "stable", "improving"]


def anthropometrics(rng, age, sex):
    """Rough Singapore-population height/weight by age and sex."""
    if age < 18:
        # simple growth curve: ~75cm at 1y to adult-ish at 17
        base_h = 75 + (age - 1) * (6.2 if age < 12 else 5.0)
        if age >= 12:
            base_h = 75 + 11 * 6.2 + (age - 12) * (5.5 if sex == "M" else 2.5)
        h = rng.gauss(base_h, 5)
        bmi = rng.gauss(16 + max(0, age - 6) * 0.35, 1.8)
    else:
        h = rng.gauss(171 if sex == "M" else 159, 6.5)
        if age > 60:
            h -= (age - 60) * 0.08
        bmi = rng.gauss(24.0 if sex == "M" else 23.0, 3.5)
    bmi = max(12.5, bmi)
    w = bmi * (h / 100) ** 2
    return round(w, 1), round(h, 1)


def pick_age(rng, lo, hi):
    # Triangular skew toward the lower-middle of the range, clipped to ints
    return int(round(rng.triangular(lo, hi, lo + (hi - lo) * 0.35)))


def ask_symptoms(rng, diagnosis, profile, symptoms):
    """Return the set of symptoms a doctor would have recorded for this visit."""
    # Everything the patient reports gets recorded
    asked = {s for s, present in symptoms.items() if present}
    if rng.random() < ASK_FEVER_P:
        asked.add("fever")
    asked |= {s for s in profile["core"] if rng.random() < ASK_CORE_P}
    # Rule out 1-2 confusable diseases
    diffs = DIFFERENTIALS[diagnosis]
    for d in rng.sample(diffs, min(len(diffs), rng.randint(1, 2))):
        asked |= {s for s in DISEASES[d]["core"] if rng.random() < ASK_DIFFERENTIAL_P}
    asked |= {s for s in profile["red_flags"] if rng.random() < ASK_RED_FLAG_P}
    return asked


def generate_row(rng, diagnosis, profile, escalate):
    age = pick_age(rng, *profile["age"])
    sex = "F" if rng.random() < profile["f_ratio"] else "M"

    # Pregnancy: only women 18-45. Pregnant patients with fever/infection
    # are a classic escalation trigger, so it is more common in escalated rows.
    pregnant = False
    if sex == "F" and 18 <= age <= 45:
        pregnant = rng.random() < (0.25 if escalate else 0.06)

    # Escalated rows lean toward high-risk ages when the disease defines one
    if escalate and rng.random() < 0.3:
        for _ in range(10):
            cand = pick_age(rng, *profile["age"])
            if profile["risk_age"](cand):
                age = cand
                if not (sex == "F" and 18 <= age <= 45):
                    pregnant = False
                break

    weight, height = anthropometrics(rng, age, sex)

    t_lo, t_hi, et_lo, et_hi = profile["temp"]
    temperature = rng.uniform(et_lo, et_hi) if escalate else rng.uniform(t_lo, t_hi)
    temperature = round(temperature, 1)

    d_lo, d_hi, ed_lo, ed_hi = profile["dur"]
    duration = rng.randint(ed_lo, ed_hi) if escalate else rng.randint(d_lo, d_hi)

    # Symptoms
    symptoms = {}
    for sid, _, _, is_red in SYMPTOMS:
        p = profile["core"].get(sid)
        if p is None:
            p = 0.0 if is_red else BACKGROUND_P
        symptoms[sid] = rng.random() < p
    symptoms["fever"] = temperature >= FEVER_THRESHOLD

    if escalate:
        # Each escalated row gets >=1 red flag (most get several);
        # ~15% rely only on risk factors (age/pregnancy/fever/duration).
        flags = profile["red_flags"]
        if rng.random() < 0.85:
            hits = [s for s, p in flags.items() if rng.random() < p]
            if not hits:
                hits = [rng.choice(list(flags))]
            for s in hits:
                symptoms[s] = True
        else:
            for s, p in flags.items():
                if not SYMPTOMS[SYMPTOM_IDS.index(s)][3]:  # non-red-flag extras only
                    symptoms[s] = rng.random() < p
        severity = rng.choices(SEVERITIES, weights=[0.05, 0.25, 0.70])[0]
        progression = rng.choices(PROGRESSIONS, weights=[0.75, 0.2, 0.05])[0]
    else:
        # Occasional label noise: a mild red-flag-like symptom without escalation
        if rng.random() < 0.05:
            symptoms[rng.choice(list(profile["red_flags"]))] = True
        severity = rng.choices(SEVERITIES, weights=[0.5, 0.42, 0.08])[0]
        progression = rng.choices(PROGRESSIONS, weights=[0.12, 0.5, 0.38])[0]

    row = {
        "age": age,
        "sex": sex,
        "pregnancy_status": pregnant,
        "temperature": temperature,
        "weight_kg": weight,
        "height_cm": height,
        "duration_days": duration,
        "severity": severity,
        "progression": progression,
    }
    asked = ask_symptoms(rng, diagnosis, profile, symptoms)
    row["symptoms"] = json.dumps({sid: symptoms[sid] for sid in SYMPTOM_IDS if sid in asked})
    row["diagnosis"] = diagnosis
    row["escalation"] = escalate
    return row


def main():
    rng = random.Random(SEED)

    with open("symptoms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symptom_id", "name", "category", "is_red_flag", "associated_diagnoses"])
        for sid, name, cat, red in SYMPTOMS:
            assoc = [d for d, p in DISEASES.items()
                     if sid in p["core"] or sid in p["red_flags"]]
            if sid == "fever":
                assoc = [d for d, p in DISEASES.items() if p["temp"][1] >= FEVER_THRESHOLD]
            w.writerow([sid, name, cat, red, "; ".join(assoc)])

    rows = []
    for diagnosis, profile in DISEASES.items():
        for escalate in (False, True):
            for _ in range(N_PER_CLASS):
                rows.append(generate_row(rng, diagnosis, profile, escalate))
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["patient_id"] = f"P{i:05d}"

    fields = ["patient_id"] + [k for k in rows[0] if k != "patient_id"]
    with open("training_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(SYMPTOMS)} symptoms and {len(rows)} training rows")


if __name__ == "__main__":
    main()
