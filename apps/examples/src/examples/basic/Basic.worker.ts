import {
  ChunkGenerator3Initializer,
  ColorArrayWithAlpha,
  createThreadedFlatWorldWorker,
  Lerp,
  LinearSpline,
  Noise,
  NOISE_TYPES,
  remap,
} from "@hello-worlds/planets"
import { Color, Vector3 } from "three"
import { interpolateColor } from "./Basic.colors"

export type ThreadParams = {
  seed: string
  terrainData: ImageData
  scaleMax: number
}

const heightGenerator: ChunkGenerator3Initializer<ThreadParams, number> = ({
  data: { seed, terrainData, scaleMax },
}) => {
  // const terrainNoise = new Noise({
  //   ...DEFAULT_NOISE_PARAMS,
  //   seed,
  //   height: 0.05,
  //   scale: 1,
  // })

  const tData: ImageData = terrainData
  const imageWidth = tData.width
  const pixelData = new Uint8Array(tData.data.buffer)
  const tempVec = new Vector3()
  const biasNoise = new Noise({
    seed,
    noiseType: NOISE_TYPES.RIGID,
    height: 2,
    scale: 10,
  })

  return ({ input, radius }) => {
    let height = 0
    let radius2 = radius * 2
    if (!tData) return height
    try {
      let x = remap(input.x, 0, radius2, 0, tData.height)
      let y = remap(input.y, 0, radius2, 0, tData.height)
      // Convert the worldCoordinates input to the image coordinate space
      tempVec.set(x, y, input.z)

      height = interpolateColor(
        tempVec.x,
        tempVec.y,
        imageWidth,
        pixelData,
        biasNoise,
      )
    } catch (e) {
      console.log(e)
    }
    const height16 = remap(height, 0, 255, 0, 1)

    // get the height from the terrain data
    return remap(height16, 0, 1, 0, scaleMax || 1000) // terrainNoise.getFromVector(input)
  }
}

const colorLerp: Lerp<THREE.Color> = (
  t: number,
  p0: THREE.Color,
  p1: THREE.Color,
) => {
  const c = p0.clone()
  return c.lerp(p1, t)
}
const colorSpline = new LinearSpline<THREE.Color>(colorLerp)
// colorSpline.addPoint(0, algea)
// colorSpline.addPoint(0.02, new Color(0xdf7126))
// colorSpline.addPoint(0.03, new Color(0xd9a066))
// colorSpline.addPoint(0.15, new Color(0xeec39a))
// colorSpline.addPoint(0.5, new Color(0x696a6a))
// colorSpline.addPoint(0.7, new Color(0x323c39))
colorSpline.addPoint(0.0, new Color(0xd9c9bb))
colorSpline.addPoint(0.01, new Color(0xe1bd9c))
colorSpline.addPoint(0.02, new Color(0x494f2b))
colorSpline.addPoint(0.1, new Color(0x6f6844))
colorSpline.addPoint(0.3, new Color(0x927e59))
colorSpline.addPoint(0.5, new Color(0x816653))
colorSpline.addPoint(0.55, new Color(0x70666d))
colorSpline.addPoint(0.6, new Color(0xffffff))

const colorGenerator: ChunkGenerator3Initializer<
  ThreadParams,
  Color | ColorArrayWithAlpha
> = ({ data: { scaleMax } }) => {
  const color = new Color(0xffffff)
  return ({ height }) => {
    const remappedHeight = remap(height, 0, scaleMax, 0, 1)
    return colorSpline.get(remappedHeight)
  }
}

createThreadedFlatWorldWorker<ThreadParams>({
  heightGenerator,
  colorGenerator,
})
