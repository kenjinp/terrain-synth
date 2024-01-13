import { MemoryRouter } from "react-router-dom"

export default function Provider({ children }) {
  return <MemoryRouter>{children}</MemoryRouter>
}
