import { useEffect, useRef, useState } from 'react'

const useRecorder = () => {
  const [recording, setRecording] = useState(false)
  const recorder = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])

  const release = () => recorder.current?.stream.getTracks().forEach((track) => track.stop())

  useEffect(() => release, [])

  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const rec = new MediaRecorder(stream)
    chunks.current = []
    rec.ondataavailable = (e) => chunks.current.push(e.data)
    rec.start()
    recorder.current = rec
    setRecording(true)
  }

  const stop = () =>
    new Promise<Blob>((resolve, reject) => {
      const rec = recorder.current
      if (!rec || rec.state === 'inactive') return reject(new Error('Not recording'))
      rec.onstop = () => {
        release()
        setRecording(false)
        resolve(new Blob(chunks.current, { type: chunks.current[0]?.type }))
      }
      rec.stop()
    })

  return { recording, start, stop }
}

export default useRecorder
