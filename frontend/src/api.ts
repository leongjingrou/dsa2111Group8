import type { Diagnosis } from './types'

export const diagnose = async (
  patientMessage: string,
  conversationId?: string,
): Promise<Diagnosis> => {
  const res = await fetch('/api/diagnose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      patient_message: patientMessage,
      conversation_id: conversationId,
    }),
  })
  if (!res.ok) throw new Error(`Diagnose failed: ${res.status}`)
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
