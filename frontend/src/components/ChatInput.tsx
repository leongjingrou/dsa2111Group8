import { useState, type FormEvent } from 'react'
import { MicIcon, StopIcon } from './Icons'
import './ChatInput.css'

type ChatInputProps = {
  onSend: (text: string) => void
  onToggleMic: () => void
  recording: boolean
  disabled: boolean
}

const ChatInput = ({ onSend, onToggleMic, recording, disabled }: ChatInputProps) => {
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
        placeholder={recording ? 'Listening...' : 'Describe your symptoms...'}
        disabled={disabled || recording}
      />
      <button
        type="button"
        className={recording ? 'mic recording' : 'mic'}
        onClick={onToggleMic}
        aria-label={recording ? 'Stop recording' : 'Speak'}
        disabled={disabled}
      >
        {recording ? <StopIcon size={20} /> : <MicIcon size={20} />}
      </button>
      <button type="submit" disabled={disabled || recording || !text.trim()}>
        Send
      </button>
    </form>
  )
}

export default ChatInput
