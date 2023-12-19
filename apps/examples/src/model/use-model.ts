import { useCallback, useEffect, useState } from "react"
import { ImageStrategy } from "./Image.strategy"
import { WGANStrategy } from "./WGAN.strategy"

export const MODEL_STRATEGIES = {
  WGAN: WGANStrategy,
  IMAGE: ImageStrategy,
}

export const MODEL_STRATEGY_NAMES = Object.keys(MODEL_STRATEGIES)

export function useModel(strategy: keyof typeof MODEL_STRATEGIES = "WGAN") {
  const [terrainModel, setTerrainModel] = useState<
    WGANStrategy | ImageStrategy
  >(() => new MODEL_STRATEGIES[strategy]())
  const [state, setState] = useState(terrainModel.state)
  const [imageData, setImageData] = useState<ImageData | null>(null)

  useEffect(() => {
    const model = new MODEL_STRATEGIES[strategy]()
    setTerrainModel(model)
    setState(model.state)
  }, [strategy])

  const load = useCallback(async () => {
    const imageData = await terrainModel.load()
    console.log({ imageData })
    setImageData(imageData)
  }, [terrainModel])
  const run = useCallback(async () => {
    const imageData = await terrainModel.run()
    setImageData(imageData)
  }, [terrainModel])

  useEffect(() => {
    terrainModel.addStateListener(s => {
      console.log(s)
      setState(s)
    })
    load()
  }, [terrainModel, load])

  return {
    state,
    imageData,
    run,
  }
}
