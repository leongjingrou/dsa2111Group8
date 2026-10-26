import { createBrowserRouter, RouterProvider } from 'react-router'
import Chat from './pages/Chat'
import Home from './pages/Home'

const router = createBrowserRouter([
  { path: '/', element: <Home /> },
  { path: '/chat', element: <Chat /> },
])

const App = () => <RouterProvider router={router} />

export default App
