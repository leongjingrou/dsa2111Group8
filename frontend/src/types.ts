export type Role = 'user' | 'bot'

export type Message = {
  role: Role
  text: string
}

export type DiagnosticStatus = 'asking' | 'complete' | 'urgent'

export type Diagnosis = {
  conversation_id: string
  status: DiagnosticStatus
  reply: string
  stop: boolean
  question_target: QuestionTarget | null
  safety: SafetyDecision
  diagnosis: ModelAnswer | null
  escalation: ModelAnswer | null
  decision_record: DecisionRecord | null
  stop_probability: number | null
  stop_threshold: number
}

export type QuestionTarget = {
  kind: 'safety' | 'symptom' | 'context' | 'semantic_clarification'
  target: string
  targets: string[] | null
  focus: string | null
  candidates?: string[]
  candidate_source?: 'word_tfidf' | 'character_vector'
}

export type RuleHit = { rule_id: string; disposition: string; reason: string }
export type SafetyDecision = {
  disposition: string
  scope_status: string
  triggered_rules: RuleHit[]
  missing_information: string[]
  policy_version: string
  event_id: string
}
export type ModelAnswer = {
  choice: string
  probabilities: Record<string, number>
  answer_confidence: number
}
export type DecisionRecord = {
  recommended_action: string
  preliminary_estimates: { condition: string; score: number | null }[]
  risk_factors: string[]
  limitations: string[]
  sources: string[]
}
export type ReviewTurn = {
  turn: number
  question_asked: string
  patient_answer: string
  response: string
  question_target: QuestionTarget | null
  answer_target: QuestionTarget | null
  extraction: {
    symptoms: { symptom_id: string; presence: string; severity: string | null; duration_days: number | null; progression: string | null }[]
    unmapped_mentions: string[]
  }
  safety: SafetyDecision
  diagnosis: ModelAnswer | null
  escalation: ModelAnswer | null
  decision_record: DecisionRecord | null
  stop_probability: number | null
  stop_threshold: number
  stop: boolean
}
export type ConsultationReview = {
  session_id: string
  finished: boolean
  pathway: ReviewTurn[]
  latest: Diagnosis | null
}
