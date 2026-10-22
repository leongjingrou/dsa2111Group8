import type { Message } from '../types'
import { BotIcon, UserIcon } from './Icons'
import './ChatBubble.css'

const ChatBubble = ({ role, text }: Message) => (
  <div className={`row ${role}`}>
    <div className="avatar">
      {role === 'user' ? <UserIcon size={18} /> : <BotIcon size={18} />}
    </div>
    <div className="bubble">{text}</div>
  </div>
)

export default ChatBubble
