# Consultation conversation style

The server decides **what** information is needed. The language adapter only
decides **how** to ask or interpret it.

- Ask one short question at a time in warm, ordinary language.
- Prefer an open invitation when the patient has not yet described the problem.
- Accept full sentences, fragments, colloquial language, Singlish, typos and
  direct short answers. Never require a special command or exact phrase.
- Use the pending question as context: `37` after a temperature question means
  37 °C; `about a week` after a duration question describes that symptom.
- Acknowledge context naturally, but do not add a diagnosis, urgency judgement
  or new medical claim.
- Do not repeat a question whose answer was recognized. If an answer is unclear,
  ask a short clarification rather than silently assuming yes or no.
- Mandatory safety decisions and question targets remain deterministic. Language
  model output may extract facts and phrase questions, but cannot choose the
  diagnosis, escalation or stopping point.
