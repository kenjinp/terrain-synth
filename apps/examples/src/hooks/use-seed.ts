import { useEffect } from "react"
import {
  createSearchParams,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router-dom"

export function useSeed() {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const seed = searchParams.get("seed")

  function setSeed(seed: string) {
    const search = createSearchParams({
      seed,
    }).toString()

    navigate({
      pathname,
      search,
    })
  }

  function setRandomSeed() {
    const newSeed = Math.random().toString(36).substring(6)
    setSeed(newSeed)
    return newSeed
  }

  useEffect(() => {
    if (!seed) {
      setRandomSeed()
    }
  }, [seed])

  return { seed, setRandomSeed } as const
}
