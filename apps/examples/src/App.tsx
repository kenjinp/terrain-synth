import { useEffect, useState } from "react"
import {
  BrowserRouter,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom"

import { Leva } from "leva"
import { Canvas } from "./components/Canvas"
import { ExampleWrapper } from "./components/ExampleWrapper"
import { Footer } from "./components/footer/Footer"
import Basic from "./examples/basic/Basic"
import { UI } from "./tunnel"

interface IRoute {
  name: string
  path: string
  component: React.ComponentType
}

const routes: IRoute[] = [
  {
    name: "Basic",
    path: "/",
    component: Basic,
  },
]

const Header = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const [currentPath, setCurrentPath] = useState(location.pathname)

  useEffect(() => {
    const name = routes.find(route => route.path === currentPath)?.name
    document.title = `Hello Worlds Examples${name ? " - " + name : ""}`
    navigate(currentPath)
  }, [currentPath])

  return (
    <header>
      {/* <a className="logo" href="https://github.com/kenjinp/hello-worlds">
        Hello Worlds
      </a>
      <select
        value={currentPath}
        onChange={event => setCurrentPath(event.target.value)}
      >
        {routes.map(route => (
          <option value={route.path} key={route.path}>
            {route.name}
          </option>
        ))}
      </select> */}
    </header>
  )
}

export default () => (
  <BrowserRouter>
    <Leva hidden />
    <Header />
    <Routes>
      {routes.map(route => (
        <Route
          path={route.path}
          key={route.path}
          element={
            <>
              <div id="ui">
                {/* Anything that goes into the tunnel, we want to render here. */}
                <UI.Out />
              </div>

              <Canvas>
                <ExampleWrapper>
                  <route.component />
                </ExampleWrapper>
              </Canvas>
            </>
          }
        />
      ))}
    </Routes>
    <Footer />
  </BrowserRouter>
)
