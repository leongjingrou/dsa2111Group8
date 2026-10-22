import type { Diagnosis, Message } from './types'

export const diagnose = async (history: Message[]): Promise<Diagnosis> => {
  const res = await fetch('/api/diagnose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: history }),
  })
  if (!res.ok) throw new Error(`Diagnose failed: ${res.status}`)
  return res.json()
}
