import { useCallback, useEffect, useState } from "react"
import { WGANStrategy } from "./WGAN.strategy"

export const MODEL_STRATEGIES = {
  WGAN: WGANStrategy,
}

export const MODEL_STRATEGY_NAMES = Object.keys(MODEL_STRATEGIES)

export function useModel(
  strategy: keyof typeof MODEL_STRATEGIES = "WGAN",
  seed?: string,
) {
  const [terrainModel, setTerrainModel] = useState<WGANStrategy>(
    () => new MODEL_STRATEGIES[strategy](),
  )
  const [state, setState] = useState(terrainModel.state)
  const [imageData, setImageData] = useState<ImageData | null>(null)

  useEffect(() => {
    const model = new MODEL_STRATEGIES[strategy]()
    setTerrainModel(model)
    setState(model.state)
  }, [strategy])

  const load = useCallback(
    async (seed?: string) => {
      const imageData = await terrainModel.load(seed)
      setImageData(imageData)
    },
    [terrainModel],
  )
  const run = useCallback(
    async (seed?: string) => {
      const imageData = await terrainModel.run(seed)
      setImageData(imageData)
    },
    [terrainModel],
  )

  useEffect(() => {
    terrainModel.addStateListener(s => {
      setState(s)
    })
    load(seed)
  }, [terrainModel, load, seed])

  return {
    state,
    imageData,
    run,
  }
}
