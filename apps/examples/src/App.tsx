import React from "react"
import { BrowserRouter, Route, Routes } from "react-router-dom"

import { Leva } from "leva"
import { Canvas } from "./components/Canvas"
import { ExampleWrapper } from "./components/ExampleWrapper"
import { Footer } from "./components/footer/Footer"
import Basic from "./pages/home/Home"
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

const hidden = false
const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Leva hidden={hidden} />
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
}

export default App
