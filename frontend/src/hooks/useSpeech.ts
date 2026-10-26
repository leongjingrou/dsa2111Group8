import { useEffect, useState } from 'react'

const supported = 'speechSynthesis' in window

const useSpeech = () => {
  const [speaking, setSpeaking] = useState<number | null>(null)

  useEffect(() => () => speechSynthesis.cancel(), [])

  const speak = (id: number, text: string) => {
    if (speechSynthesis.speaking || speechSynthesis.pending) speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.onend = utterance.onerror = () => setSpeaking((cur) => (cur === id ? null : cur))
    speechSynthesis.speak(utterance)
    setSpeaking(id)
  }

  const stop = () => {
    speechSynthesis.cancel()
    setSpeaking(null)
  }

  return { speaking, speak, stop, supported }
}

export default useSpeech
