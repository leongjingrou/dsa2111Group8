import { createBrowserRouter, RouterProvider } from 'react-router'
import Chat from './pages/Chat'
import Home from './pages/Home'
import Doctor from './pages/Doctor'

const router = createBrowserRouter([
  { path: '/', element: window.location.port === '8081' ? <Doctor /> : <Home /> },
  { path: '/chat', element: <Chat /> },
  { path: '/doctor', element: <Doctor /> },
  { path: '/doctor/:sessionId', element: <Doctor /> },
])

const App = () => <RouterProvider router={router} />

export default App
