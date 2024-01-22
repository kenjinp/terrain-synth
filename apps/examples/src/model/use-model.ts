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
  const [result, setImageData] = useState<{
    terrainData: Uint8Array
    oceanData: Uint8Array
  } | null>(null)

  useEffect(() => {
    const model = new MODEL_STRATEGIES[strategy]()
    setTerrainModel(model)
    setState(model.state)
    console.log("NEW MODEL")
    return () => {
      console.log("DESTROY MODEL")
      model.worker.terminate()
    }
  }, [strategy])

  const load = useCallback(
    async (seed?: string) => {
      try {
        const result = await terrainModel.load(seed)
        setImageData(result)
      } catch (err) {
        console.error(err)
      }
    },
    [terrainModel],
  )
  const run = useCallback(
    async (seed?: string) => {
      try {
        const result = await terrainModel.run(seed)
        setImageData(result)
      } catch (err) {
        console.error(err)
      }
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
    result,
    run,
  }
}
