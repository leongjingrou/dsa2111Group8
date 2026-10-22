import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { diagnose } from '../api'
import ChatBubble from '../components/ChatBubble'
import ChatInput from '../components/ChatInput'
import { BotIcon } from '../components/Icons'
import type { DiagnosticStatus, Message } from '../types'
import './Chat.css'

const greeting: Message = {
  role: 'bot',
  text: 'Hello! I am your telemedicine assistant. How are you feeling today?',
}

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([greeting])
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<DiagnosticStatus>('asking')
  const navigate = useNavigate()
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = async (text: string) => {
    const history: Message[] = [...messages, { role: 'user', text }]
    setMessages(history)
    setLoading(true)
    try {
      const result = await diagnose(history)
      setMessages((prev) => [...prev, { role: 'bot', text: result.reply }])
      setStatus(result.status)
    } catch {
      setMessages((prev) => [...prev, { role: 'bot', text: 'Sorry, something went wrong. Please try again.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="chat">
      <header>
        <BotIcon size={22} />
        <span>Telemedicine Assistant</span>
      </header>
      <section className="messages">
        {messages.map((msg, i) => (
          <ChatBubble key={i} {...msg} />
        ))}
        {loading && <ChatBubble role="bot" text="..." />}
        <div ref={bottom} />
      </section>
      {status !== 'asking' && (
        <button className="end-diagnosis" onClick={() => navigate('/')}>
          End consultation
        </button>
      )}
      <ChatInput onSend={send} onMic={() => {}} disabled={loading} />
    </main>
  )
}

export default Chat
