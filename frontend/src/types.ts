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
}
