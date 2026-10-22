import { useState, type FormEvent } from 'react'
import { MicIcon } from './Icons'
import './ChatInput.css'

type ChatInputProps = {
  onSend: (text: string) => void
  onMic: () => void
}

const ChatInput = ({ onSend, onMic }: ChatInputProps) => {
  const [text, setText] = useState('')

  const submit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const msg = text.trim()
    if (!msg) return
    onSend(msg)
    setText('')
  }

  return (
    <form className="chat-input" onSubmit={submit}>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Describe your symptoms..."
      />
      <button type="button" className="mic" onClick={onMic} aria-label="Speak">
        <MicIcon size={20} />
      </button>
      <button type="submit">Send</button>
    </form>
  )
}

export default ChatInput
