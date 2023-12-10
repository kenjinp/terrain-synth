import { useCallback, useEffect, useState } from "react"
import { ThreadedTerrainSynthModel } from "./Model"

export function useModel() {
  const [terrainModel] = useState(() => new ThreadedTerrainSynthModel())
  const [state, setState] = useState(terrainModel.state)
  const [imageData, setImageData] = useState<ImageData | null>(null)

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
