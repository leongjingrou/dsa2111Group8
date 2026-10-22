import { useNavigate } from 'react-router'
import { BotIcon } from '../components/Icons'
import './Home.css'

const Home = () => {
  const navigate = useNavigate()

  return (
    <main className="home">
      <div className="card">
        <div className="avatar large">
          <BotIcon size={40} />
        </div>
        <h1>Telemedicine Assistant</h1>
        <p>Describe your symptoms by text or voice and get a preliminary assessment.</p>
        <button onClick={() => navigate('/chat')}>Start chat</button>
      </div>
    </main>
  )
}

export default Home
