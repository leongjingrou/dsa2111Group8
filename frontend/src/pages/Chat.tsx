import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { diagnose, transcribe } from '../api'
import ChatBubble from '../components/ChatBubble'
import ChatInput from '../components/ChatInput'
import { BackIcon, BotIcon } from '../components/Icons'
import useRecorder from '../hooks/useRecorder'
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
  const { recording, start, stop } = useRecorder()
  const navigate = useNavigate()
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const reply = (text: string) => setMessages((prev) => [...prev, { role: 'bot', text }])

  const toggleMic = async () => {
    if (!recording) return start().catch(() => reply('Unable to access the microphone. Please check your permissions.'))
    setLoading(true)
    const text = (await stop().then(transcribe).catch(() => '')).trim()
    setLoading(false)
    if (text) send(text)
    else reply("Sorry, I couldn't catch that. Please try again.")
  }

  const send = async (text: string) => {
    const history: Message[] = [...messages, { role: 'user', text }]
    setMessages(history)
    setLoading(true)
    try {
      const result = await diagnose(history)
      reply(result.reply)
      setStatus(result.status)
    } catch {
      reply('Sorry, something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="chat">
      <header>
        <button className="back" onClick={() => navigate('/')} aria-label="Back to home">
          <BackIcon size={20} />
        </button>
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
      <ChatInput onSend={send} onToggleMic={toggleMic} recording={recording} disabled={loading} />
    </main>
  )
}

export default Chat
