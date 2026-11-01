import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { diagnose, SafetyAuditError, transcribe } from '../api'
import ChatBubble from '../components/ChatBubble'
import ChatInput from '../components/ChatInput'
import { BackIcon } from '../components/Icons'
import useRecorder from '../hooks/useRecorder'
import useSpeech from '../hooks/useSpeech'
import type { DiagnosticStatus, Message } from '../types'
import './Chat.css'

const greeting: Message = {
  role: 'bot',
  text: 'Hello, what seems to be the problem today?',
}

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([greeting])
  const [conversationId, setConversationId] = useState<string>()
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<DiagnosticStatus>('asking')
  const { recording, start, stop } = useRecorder()
  const { speaking, speak, stop: silence, supported } = useSpeech()
  const navigate = useNavigate()
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const reply = (text: string) => setMessages((prev) => [...prev, { role: 'bot', text }])

  const toggleMic = async () => {
    if (!recording) {
      silence()
      return start().catch(() => reply('Unable to access the microphone. Please check your permissions.'))
    }
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
      const result = await diagnose(text, conversationId, crypto.randomUUID())
      setConversationId(result.conversation_id)
      reply(result.reply)
      setStatus(result.status)
    } catch (cause) {
      if (cause instanceof SafetyAuditError) setStatus('urgent')
      reply(cause instanceof Error && !cause.message.startsWith('Diagnose failed:')
        ? cause.message
        : 'Sorry, something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const openDoctorView = () => {
    if (!conversationId) return
    const url = new URL(`/doctor/${encodeURIComponent(conversationId)}`, window.location.href)
    if (url.port === '8080') url.port = '8081'
    window.open(url.toString(), '_blank', 'noopener,noreferrer')
  }

  return (
    <main className="chat">
      <header className="chat-header">
        <button className="back" onClick={() => navigate('/')} aria-label="Back to home">
          <BackIcon size={20} />
        </button>
        <span className="chat-brand"><span className="brand-mark">+</span><span><strong>Clinical intake</strong><small>Patient consultation</small></span></span>
        <span className="chat-state">{status === 'asking' ? 'In progress' : 'Completed'}</span>
        {conversationId && <button className="doctor-link" onClick={openDoctorView}>Doctor view</button>}
      </header>
      <div className="chat-context"><strong>Symptom assessment</strong><span>Answer in your own words. Brief replies are okay when they answer the latest question.</span></div>
      <section className="messages">
        {messages.map((msg, i) => (
          <ChatBubble
            key={i}
            {...msg}
            speaking={speaking === i}
            onSpeak={msg.role === 'bot' && supported ? () => (speaking === i ? silence() : speak(i, msg.text)) : undefined}
          />
        ))}
        {loading && <ChatBubble role="bot" text="..." />}
        <div ref={bottom} />
      </section>
      {status !== 'asking' && (
        <div className="chat-finish"><span>Assessment complete</span><button className="end-diagnosis" onClick={() => navigate('/')}>Return home</button></div>
      )}
      {status === 'asking' && (
        <ChatInput onSend={send} onToggleMic={toggleMic} recording={recording} disabled={loading} />
      )}
    </main>
  )
}

export default Chat
