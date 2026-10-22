import type { Message } from '../types'
import { BotIcon, SpeakerIcon, StopIcon, UserIcon } from './Icons'
import './ChatBubble.css'

type Props = Message & {
  speaking?: boolean
  onSpeak?: () => void
}

const ChatBubble = ({ role, text, speaking, onSpeak }: Props) => (
  <div className={`row ${role}`}>
    <div className="avatar">
      {role === 'user' ? <UserIcon size={18} /> : <BotIcon size={18} />}
    </div>
    <div className="bubble">{text}</div>
    {onSpeak && (
      <button className="speak" onClick={onSpeak} aria-label={speaking ? 'Stop reading' : 'Read aloud'}>
        {speaking ? <StopIcon size={18} /> : <SpeakerIcon size={18} />}
      </button>
    )}
  </div>
)

export default ChatBubble
