import type { Message } from '../types'
import { SpeakerIcon, StopIcon } from './Icons'
import './ChatBubble.css'

type Props = Message & {
  speaking?: boolean
  onSpeak?: () => void
}

const ChatBubble = ({ role, text, speaking, onSpeak }: Props) => (
  <div className={`row ${role}`}>
    <div className="bubble-wrap"><span className="bubble-author">{role === 'user' ? 'You' : 'Intake'}</span><div className="bubble">{text}</div></div>
    {onSpeak && (
      <button className="speak" onClick={onSpeak} aria-label={speaking ? 'Stop reading' : 'Read aloud'}>
        {speaking ? <StopIcon size={18} /> : <SpeakerIcon size={18} />}
      </button>
    )}
  </div>
)

export default ChatBubble
