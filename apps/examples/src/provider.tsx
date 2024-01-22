import { MemoryRouter } from "react-router-dom"

export default function Provider({ children }: { children: React.ReactNode }) {
  return <MemoryRouter>{children}</MemoryRouter>
}
