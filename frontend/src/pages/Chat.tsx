import { useEffect, useRef, useState } from 'react'
import ChatBubble from '../components/ChatBubble'
import ChatInput from '../components/ChatInput'
import { BotIcon } from '../components/Icons'
import type { Message } from '../types'
import './Chat.css'

const greeting: Message = {
  role: 'bot',
  text: 'Hello! I am your telemedicine assistant. How are you feeling today?',
}

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([greeting])
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = (text: string) => setMessages((prev) => [...prev, { role: 'user', text }])

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
        <div ref={bottom} />
      </section>
      <ChatInput onSend={send} onMic={() => {}} />
    </main>
  )
}

export default Chat
