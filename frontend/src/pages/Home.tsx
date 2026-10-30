import { useNavigate } from 'react-router'
import './Home.css'

const Home = () => {
  const navigate = useNavigate()

  return (
    <main className="home">
      <div className="home-shell">
        <div className="home-brand"><span className="brand-mark">+</span><span>Clinical intake</span><span className="home-label">DEMO</span></div>
        <div className="home-content">
          <p className="home-kicker">PATIENT INTAKE</p>
          <h1>Tell us what is going on.</h1>
          <p className="home-lead">Answer a few focused questions about your symptoms. You can type or speak, and you will see a preliminary next step when the intake is complete.</p>
          <button className="home-start" onClick={() => navigate('/chat')}>Begin consultation <span aria-hidden="true">→</span></button>
          <p className="home-caution">For emergencies, seek urgent medical help. This tool is a prototype and does not replace a clinician.</p>
        </div>
        <div className="home-footer"><span>Structured symptom review</span><span>Safety screening included</span></div>
      </div>
    </main>
  )
}

export default Home
