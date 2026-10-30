import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'
import { getConsultationReview } from '../api'
import type { ConsultationReview, ModelAnswer, ReviewTurn } from '../types'
import './Doctor.css'

const percentage = (value: number) => `${(value * 100).toFixed(1)}%`
const label = (value: string) => value.replaceAll('_', ' ')

function ModelStats({ title, answer }: { title: string; answer: ModelAnswer | null }) {
  if (!answer) return <p className="doctor-muted">{title}: no model result at this step.</p>
  return (
    <div className="doctor-model">
      <strong>{title}: {label(answer.choice)}</strong>
      <span>Confidence {percentage(answer.answer_confidence)}</span>
      <div className="doctor-probabilities">
        {Object.entries(answer.probabilities)
          .sort((a, b) => b[1] - a[1])
          .map(([name, value]) => (
            <div className="doctor-probability" key={name}>
              <span>{label(name)}</span><meter min="0" max="1" value={value} />
              <span>{percentage(value)}</span>
            </div>
          ))}
      </div>
    </div>
  )
}

function PathwayStep({ step }: { step: ReviewTurn }) {
  const rules = step.safety.triggered_rules
  return (
    <li className="doctor-step">
      <div className="doctor-step-title">
        <strong>Turn {step.turn}</strong>
        <span className={`doctor-pill ${step.safety.disposition}`}>{label(step.safety.disposition)}</span>
      </div>
      <dl>
        <dt>Question asked</dt><dd>{step.question_asked}</dd>
        <dt>Patient answer</dt><dd>{step.patient_answer}</dd>
        <dt>Answer applied to</dt><dd>{step.answer_target ? `${label(step.answer_target.target)}${step.answer_target.focus ? ` · ${label(step.answer_target.focus)}` : ''}` : 'Opening description'}</dd>
        <dt>Next response</dt><dd>{step.response}</dd>
        <dt>Next question</dt><dd>{step.question_target ? `${label(step.question_target.target)}${step.question_target.focus ? ` · ${label(step.question_target.focus)}` : ''}` : 'Consultation complete'}</dd>
      </dl>
      {(step.extraction.symptoms.length > 0 || step.extraction.unmapped_mentions.length > 0) && <div className="doctor-reasons"><strong>Interpreted facts</strong><ul>{step.extraction.symptoms.map((fact, index) => <li key={`${fact.symptom_id}-${index}`}>{label(fact.symptom_id)}: {fact.presence}{fact.severity ? `, ${fact.severity}` : ''}{fact.duration_days !== null ? `, ${fact.duration_days} day(s)` : ''}{fact.progression ? `, ${fact.progression}` : ''}</li>)}{step.extraction.unmapped_mentions.map((mention, index) => <li key={`unmapped-${index}`}>Unmapped: {mention}</li>)}</ul></div>}
      {rules.length > 0 && <div className="doctor-reasons"><strong>Rules fired</strong><ul>{rules.map((rule) => <li key={rule.rule_id}>{rule.rule_id}: {rule.reason}</li>)}</ul></div>}
      {step.safety.missing_information.length > 0 && <p><strong>Missing:</strong> {step.safety.missing_information.map(label).join(', ')}</p>}
      {step.diagnosis && <details><summary>Model outputs at this step</summary><ModelStats title="Condition" answer={step.diagnosis} /><ModelStats title="Escalation" answer={step.escalation} /><p>Stop probability {step.stop_probability === null ? 'unavailable' : percentage(step.stop_probability)} · threshold {percentage(step.stop_threshold)}</p></details>}
    </li>
  )
}

export default function Doctor() {
  const { sessionId } = useParams()
  const navigate = useNavigate()
  const [enteredId, setEnteredId] = useState('')
  const [review, setReview] = useState<ConsultationReview | null>(null)
  const [error, setError] = useState('')
  const load = useCallback(async () => {
    if (!sessionId) return
    try {
      setReview(await getConsultationReview(sessionId))
      setError('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to load consultation.')
    }
  }, [sessionId])

  useEffect(() => {
    const initial = window.setTimeout(() => void load(), 0)
    const timer = window.setInterval(() => void load(), 3000)
    return () => { window.clearTimeout(initial); window.clearInterval(timer) }
  }, [load])

  const latest = review?.latest
  return (
    <main className="doctor">
      <header className="doctor-header">
        <div className="doctor-heading"><span className="brand-mark">+</span><div><p className="doctor-eyebrow">CLINICIAN WORKSPACE · LOCAL DEMO</p><h1>Consultation review</h1></div></div>
        <div className="doctor-actions"><span className="doctor-live">Syncs every 3 seconds</span><button onClick={() => void load()}>Refresh</button><Link to="/">Home</Link></div>
      </header>
      <p className="doctor-note">Recorded patient answers, question path, safety decisions and model estimates. For review by a clinician; this is a preliminary assessment.</p>
      {!sessionId && <form className="doctor-lookup" onSubmit={(event) => { event.preventDefault(); if (enteredId.trim()) navigate(`/doctor/${encodeURIComponent(enteredId.trim())}`) }}><label htmlFor="session-id">Consultation ID</label><input id="session-id" value={enteredId} onChange={(event) => setEnteredId(event.target.value)} placeholder="Paste the ID from the consultation" /><button type="submit">Open review</button></form>}
      {error && <p className="doctor-error" role="alert">{error}</p>}
      {sessionId && !review && !error && <p>Loading consultation…</p>}
      {review && <>
        <div className="doctor-record"><span>CONSULTATION RECORD</span><code>{review.session_id}</code><span className={review.finished ? 'doctor-record-state done' : 'doctor-record-state'}>{review.finished ? 'Completed' : 'In progress'}</span></div>
        <div className="doctor-summary">
          <section><span>Status</span><strong>{latest ? label(latest.status) : 'waiting for first answer'}</strong></section>
          <section><span>Answers</span><strong>{review.pathway.length}</strong></section>
          <section><span>Safety disposition</span><strong>{latest ? label(latest.safety.disposition) : 'pending'}</strong></section>
          <section><span>Recommended action</span><strong>{latest?.decision_record ? label(latest.decision_record.recommended_action) : 'pending'}</strong></section>
        </div>
        <div className="doctor-grid">
          <section className="doctor-panel">
            <h2>Question pathway</h2>
            {review.pathway.length ? <ol className="doctor-pathway">{review.pathway.map((step) => <PathwayStep key={step.turn} step={step} />)}</ol> : <p>No patient answers yet.</p>}
          </section>
          <aside className="doctor-panel">
            <h2>Latest assessment</h2>
            {latest ? <>
              <p><strong>Safety:</strong> {label(latest.safety.disposition)} · {label(latest.safety.scope_status)}</p>
              <p><strong>Policy:</strong> {latest.safety.policy_version}</p>
              <p><strong>Audit event:</strong> <code>{latest.safety.event_id}</code></p>
              <ModelStats title="Condition" answer={latest.diagnosis} />
              <ModelStats title="Escalation" answer={latest.escalation} />
              <p><strong>Stop probability:</strong> {latest.stop_probability === null ? 'unavailable' : percentage(latest.stop_probability)} · threshold {percentage(latest.stop_threshold)}</p>
              {latest.decision_record && <><h3>Decision record</h3><p>Action: {label(latest.decision_record.recommended_action)}</p>{latest.decision_record.risk_factors.length > 0 && <p>Risk factors: {latest.decision_record.risk_factors.join(', ')}</p>}{latest.decision_record.limitations.length > 0 && <p>Limits: {latest.decision_record.limitations.join('; ')}</p>}</>}
            </> : <p>Assessment starts after the first patient answer.</p>}
          </aside>
        </div>
      </>}
    </main>
  )
}
