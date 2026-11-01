import type { ConsultationReview, Diagnosis } from './types'

export class SafetyAuditError extends Error {}

export const diagnose = async (
  patientMessage: string,
  conversationId?: string,
  requestId?: string,
): Promise<Diagnosis> => {
  const res = await fetch('/api/diagnose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      patient_message: patientMessage,
      conversation_id: conversationId,
      request_id: requestId,
    }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    if (res.status === 503 && body?.detail?.reply) throw new SafetyAuditError(body.detail.reply)
    throw new Error(`Diagnose failed: ${res.status}`)
  }
  return res.json()
}

export const getConsultationReview = async (sessionId: string): Promise<ConsultationReview> => {
  const res = await fetch(`/api/conversations/${encodeURIComponent(sessionId)}/review`, {
    cache: 'no-store',
  })
  if (!res.ok) throw new Error(res.status === 404 ? 'Consultation not found or expired.' : 'Unable to load consultation.')
  return res.json()
}

export const transcribe = async (audio: Blob): Promise<string> => {
  const ext = audio.type.split(/[/;]/)[1]
  const form = new FormData()
  form.append('audio', audio, `speech.${ext}`)
  const res = await fetch('/api/transcribe', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Transcribe failed: ${res.status}`)
  const { text }: { text: string } = await res.json()
  return text
}
