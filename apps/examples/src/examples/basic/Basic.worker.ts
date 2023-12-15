import {
  ChunkGenerator3Initializer,
  ColorArrayWithAlpha,
  NOISE_TYPES,
  Noise,
  createThreadedFlatWorldWorker,
  remap,
} from "@hello-worlds/planets"
import { Color, MathUtils, Vector3 } from "three"
import { BIOMES, biomeColorSplineMap, oceanSpline } from "./Basic.biomes"
import { getPixelColor, interpolateColor } from "./Basic.colors"

export type ThreadParams = {
  seed: string
  terrainData: ImageData
  oceanData: ImageData
  scaleMax: number
  biome: BIOMES
  useNoise: boolean
  useInterpolation: boolean
}

const heightGenerator: ChunkGenerator3Initializer<ThreadParams, number> = ({
  data: { seed, terrainData, oceanData, scaleMax, useNoise, useInterpolation },
}) => {
  const tData: ImageData = terrainData
  const imageWidth = tData.width
  const pixelData = new Uint8Array(tData.data.buffer)
  const tempVec = new Vector3()

  const oData: ImageData = oceanData
  const oceanPixelData = new Uint8Array(oData.data.buffer)
  const oceanNoise = new Noise({
    seed,
    noiseType: NOISE_TYPES.BILLOWING,
    height: 300,
    scale: scaleMax * 2,
  })

  return ({ input, radius }) => {
    let height = 0
    const halfSize = radius
    if (!tData) return height
    let x = remap(input.x, -halfSize, halfSize, 0, tData.height - 1)
    let y = remap(input.y, -halfSize, halfSize, 0, tData.height - 1)

    // Convert the worldCoordinates input to the image coordinate space
    tempVec.set(x, y, input.z)

    if (useInterpolation) {
      height = interpolateColor(
        tempVec.x,
        tempVec.y,
        imageWidth,
        pixelData,
        // biasNoise,
        useNoise,
      )
    } else {
      height = getPixelColor(
        MathUtils.clamp(Math.floor(tempVec.x), 0, imageWidth - 1),
        MathUtils.clamp(Math.floor(tempVec.y), 0, imageWidth - 1),
        imageWidth,
        pixelData,
      )
    }
    // sea level
    if (height <= 0) {
      if (useInterpolation) {
        height = interpolateColor(
          tempVec.x,
          tempVec.y,
          imageWidth,
          oceanPixelData,
          // biasNoise,
          useNoise,
        )
      } else {
        height = getPixelColor(
          MathUtils.clamp(Math.floor(tempVec.x), 0, imageWidth - 1),
          MathUtils.clamp(Math.floor(tempVec.y), 0, imageWidth - 1),
          imageWidth,
          oceanPixelData,
        )
      }
      const height64 = remap(height, 0, 255, 0, 1)
      let oceanHeight = -remap(height64, 0, 1, 0, scaleMax * 4)
      const maxDepth = -scaleMax + 100
      //  add more noise the deeper it gets
      const o = oceanNoise.getFromVector(input)
      oceanHeight = MathUtils.lerp(oceanHeight, oceanHeight + o, 0.5)

      return MathUtils.clamp(oceanHeight, maxDepth, 0)
    }
    const height64 = remap(height, 0, 255, 0, 1)

    // get the height from the terrain data
    return remap(height64, 0, 1, 0, scaleMax || 1000) // terrainNoise.getFromVector(input)
  }
}

const colorGenerator: ChunkGenerator3Initializer<
  ThreadParams,
  Color | ColorArrayWithAlpha
> = ({ data: { scaleMax, biome } }) => {
  const colorSpline =
    biomeColorSplineMap[biome] || biomeColorSplineMap[BIOMES.SIM_CITY]
  return ({ height }) => {
    if (height < 5) {
      return oceanSpline.get(remap(height, -100, 0, 1, 0))
    }
    const remappedHeight = remap(height, 0, scaleMax, 0, 1)
    return colorSpline.get(remappedHeight)
  }
}

createThreadedFlatWorldWorker<ThreadParams>({
  heightGenerator,
  colorGenerator,
})
