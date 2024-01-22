import {
  Chunk,
  DEFAULT_NOISE_PARAMS,
  Noise,
  NoiseParams,
  remap,
} from "@hello-worlds/planets"
import PoissonDiskSampling from "poisson-disk-sampling"
import { Vector3 } from "three"
import { interpolateColor } from "../../pages/home/Home.colors"

function generateSlopMask() {}

export interface SlopeMaskConfig {
  min: number
  max: number
}

export interface HeightMaskConfig {
  min: number
  max: number
}

export interface DensityMapConfig {
  minDistance: number
  maxDistance: number
  noiseConfig: NoiseParams
}

export const defaultDensityMapConfig: DensityMapConfig = {
  minDistance: 0,
  maxDistance: 100,
  noiseConfig: {
    ...DEFAULT_NOISE_PARAMS,
    height: 1,
  },
}

export interface ScatterMapConfig {
  heightMask?: HeightMaskConfig
  slopeMask?: SlopeMaskConfig
  densityMap?: DensityMapConfig
}

export class ScatterMap {
  slopeMask?: Uint8Array
  heightMask?: Uint8Array
  densityMap?: Uint8Array
  scatterMap: Uint8Array
  points?: number[][]
  constructor(
    public seed: string,
    public heightmap: ArrayBuffer,
    minHeight: number,
    maxHeight: number,
    public scatterMapConfig: ScatterMapConfig,
  ) {
    const {
      heightMask,
      slopeMask,
      densityMap = defaultDensityMapConfig,
    } = scatterMapConfig

    const scatterMapData = new Uint8Array(heightmap.byteLength)
    const heightmapData = new Uint8Array(heightmap)
    const heightmapWidth = Math.sqrt(heightmapData.length / 4)

    function generateSlopeMap() {
      // lets create a slope map of the heightmap
      const slopeMapData = new Uint8Array(heightmap.byteLength)
      const slopeMapWidth = Math.sqrt(slopeMapData.length / 4)
      for (let x = 0; x < slopeMapWidth; x++) {
        for (let y = 0; y < slopeMapWidth; y++) {
          const index = (x + y * slopeMapWidth) * 4
          const height = remap(
            heightmapData[index],
            0,
            255,
            minHeight,
            maxHeight,
          )
          const heightRight = remap(
            heightmapData[index + 4],
            0,
            255,
            minHeight,
            maxHeight,
          )
          const heightDown = remap(
            heightmapData[index + slopeMapWidth * 4],
            0,
            255,
            minHeight,
            maxHeight,
          )
          const heightLeft = remap(
            heightmapData[index - 4],
            0,
            255,
            minHeight,
            maxHeight,
          )
          const heightUp = remap(
            heightmapData[index - slopeMapWidth * 4],
            0,
            255,
            minHeight,
            maxHeight,
          )

          const slopeRight = Math.abs(height - heightRight)
          const slopeDown = Math.abs(height - heightDown)
          const slopeLeft = Math.abs(height - heightLeft)
          const slopeUp = Math.abs(height - heightUp)

          const slope = Math.max(slopeRight, slopeDown, slopeLeft, slopeUp)

          slopeMapData[index] = slope
          slopeMapData[index + 1] = slope
          slopeMapData[index + 2] = slope
          slopeMapData[index + 3] = 255
        }
      }
      return slopeMapData
    }

    function generateSlopeMask(
      slopeMask: SlopeMaskConfig,
      slopeMapData: Uint8Array,
    ) {
      // lets create a slope map of the heightmap
      const slopeMaskData = new Uint8Array(heightmap.byteLength)
      const slopeMaskWidth = Math.sqrt(slopeMapData.length / 4)
      const { min, max } = slopeMask
      for (let x = 0; x < slopeMaskWidth; x++) {
        for (let y = 0; y < slopeMaskWidth; y++) {
          const index = (x + y * slopeMaskWidth) * 4
          const slope = slopeMapData[index]
          if (slope < min || slope > max) {
            slopeMaskData[index] = 0
            slopeMaskData[index + 1] = 0
            slopeMaskData[index + 2] = 0
            slopeMaskData[index + 3] = 0
          } else {
            slopeMaskData[index] = 255
            slopeMaskData[index + 1] = 255
            slopeMaskData[index + 2] = 255
            slopeMaskData[index + 3] = 255
          }
        }
      }
      return slopeMaskData
    }

    function generateHeightMask(heightMask: HeightMaskConfig) {
      // lets create a slope map of the heightmap
      const heightMaskData = new Uint8Array(heightmap.byteLength)
      const { min, max } = heightMask
      for (let x = 0; x < heightmapWidth; x++) {
        for (let y = 0; y < heightmapWidth; y++) {
          const index = (x + y * heightmapWidth) * 4
          const height = remap(
            heightmapData[index],
            0,
            255,
            minHeight,
            maxHeight,
          )
          if (height < min || height > max) {
            heightMaskData[index] = 0
            heightMaskData[index + 1] = 0
            heightMaskData[index + 2] = 0
            heightMaskData[index + 3] = 255
          } else {
            heightMaskData[index] = 255
            heightMaskData[index + 1] = 255
            heightMaskData[index + 2] = 255
            heightMaskData[index + 3] = 255
          }
        }
      }
      return heightMaskData
    }

    function generateDensityMap(densityMap: DensityMapConfig) {
      const { minDistance, maxDistance, noiseConfig } = densityMap
      const densityMapData = new Uint8Array(heightmap.byteLength)
      const densityMapWidth = Math.sqrt(densityMapData.length / 4)
      const noise = new Noise(noiseConfig)
      for (let x = 0; x < densityMapWidth; x++) {
        for (let y = 0; y < densityMapWidth; y++) {
          const index = (x + y * densityMapWidth) * 4
          const density = remap(
            noise.get(x, 0, y),
            -noiseConfig.height,
            noiseConfig.height,
            0,
            255,
          )
          densityMapData[index] = density
          densityMapData[index + 1] = density
          densityMapData[index + 2] = density
          densityMapData[index + 3] = 255
        }
      }
      return densityMapData
    }

    if (slopeMask) {
      const slopeMapData = generateSlopeMap()
      this.slopeMask = generateSlopeMask(slopeMask, slopeMapData)
    }

    if (heightMask) {
      this.heightMask = generateHeightMask(heightMask)
    }

    this.densityMap = generateDensityMap(densityMap)

    // lets create a scatter map by multiplying the slope map by the height mask
    // and then multiplying the result by the density map

    function multiply(a: Uint8Array, b: Uint8Array) {
      const result = new Uint8Array(heightmap.byteLength)
      for (let i = 0; i < heightmap.byteLength; i++) {
        const aVal = remap(a[i], 0, 255, 0, 1)
        const bVal = remap(b[i], 0, 255, 0, 1)
        result[i] = remap(aVal * bVal, 0, 1, 0, 255)
      }
      return result
    }

    scatterMapData.set(this.densityMap)

    if (this.heightMask) {
      scatterMapData.set(multiply(scatterMapData, this.heightMask))
    }
    if (this.slopeMask) {
      scatterMapData.set(multiply(scatterMapData, this.slopeMask))
    }

    this.scatterMap = scatterMapData
  }

  generatePoints(chunk: Chunk, limit = 1000) {
    const tempVec3 = new Vector3()
    const p = new PoissonDiskSampling({
      shape: [chunk.width, chunk.width],
      minDistance: 20,
      maxDistance: 500,
      tries: 10,
      // distanceFunction: point => {
      //   // read value from scatter map
      //   const x = Math.floor(point[0])
      //   const y = Math.floor(point[1])
      //   const index = (x + y * chunk.width) * 4
      //   const value = this.scatterMap[index]
      //   const alpha = remap(value, 0, 255, 0, 1)
      //   return alpha
      // },
    })
    let points = p.fill()
    // sample max 10k points
    points.length = Math.min(points.length, limit)
    points = points.slice(0, limit)

    const u8 = new Uint8Array(this.heightmap)
    const subOffset = new Vector3(chunk.offset.x, 0, chunk.offset.y)

    for (let index = 0; index < points.length; index++) {
      const point = points[index]

      tempVec3.set(point[0], 0, point[1]).sub(subOffset)

      // read the y value from the heightmap
      const y = interpolateColor(
        point[0],
        point[1],
        Math.sqrt(this.heightmap.byteLength / 4),
        u8,
      )
      tempVec3.setY(remap(y, 0, 255, chunk.minHeight, chunk.maxHeight))

      points[index] = tempVec3.toArray()
    }
    return points
  }
}
