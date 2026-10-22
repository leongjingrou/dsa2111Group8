export type Role = 'user' | 'bot'

export type Message = {
  role: Role
  text: string
}
