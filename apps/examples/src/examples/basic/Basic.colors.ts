import { NOISE_TYPES, Noise } from "@hello-worlds/planets"
import { MathUtils } from "three"

const biasNoise = new Noise({
  seed: "banana",
  noiseType: NOISE_TYPES.RIGID,
  height: 2,
  scale: 10,
})

const warpNoise = new Noise({
  seed: "banana",
  noiseType: NOISE_TYPES.BILLOWING,
  height: 4,
  scale: 1000,
})

export function interpolateColor(
  x: number,
  y: number,
  width: number,
  pixelData: Uint8Array,
  // noise: Noise,
  useNoise = true,
) {
  // lets push the noise from the origin, which might otherwise produce artifacts
  const noiseOffset = 1000
  let n = biasNoise.get(x + noiseOffset, 0, y + noiseOffset)
  let e = warpNoise.get(x + noiseOffset + n, 0, y + n + noiseOffset)
  let f = biasNoise.get(x + noiseOffset + e, 0, y + e + noiseOffset)
  let g = warpNoise.get(x + noiseOffset + f, 0, y + f + noiseOffset)
  if (useNoise) {
    x = x + n + e + f + g
    y = y + n + e + f + g
  }

  // Get the integer and fractional parts of the coordinates
  const x0 = MathUtils.clamp(Math.floor(x), 0, width - 1)
  const y0 = MathUtils.clamp(Math.floor(y), 0, width - 1)
  const x1 = MathUtils.clamp(x0 + 1, 0, width - 1)
  const y1 = MathUtils.clamp(y0 + 1, 0, width - 1)

  // Ensure the coordinates are within the image bounds
  if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= width) {
    throw new Error("Interpolation coordinates are outside the image bounds")
  }

  // Get the colors of the four surrounding pixels
  const topLeft = getPixelColor(x0, y0, width, pixelData)
  const topRight = getPixelColor(x1, y0, width, pixelData)
  const bottomLeft = getPixelColor(x0, y1, width, pixelData)
  const bottomRight = getPixelColor(x1, y1, width, pixelData)

  // Calculate the weights based on the fractional parts
  const weightX = x - x0
  const weightY = y - y0

  return interpolateChannel(
    topLeft,
    topRight,
    bottomLeft,
    bottomRight,
    weightX,
    weightY,
  )
}

function interpolateChannel(
  c00: number,
  c01: number,
  c10: number,
  c11: number,
  weightX: number,
  weightY: number,
) {
  // Bilinear interpolation formula
  return (
    c00 * (1 - weightX) * (1 - weightY) +
    c01 * weightX * (1 - weightY) +
    c10 * (1 - weightX) * weightY +
    c11 * weightX * weightY
  )
}

export function getPixelColor(
  x: number,
  y: number,
  width: number,
  pixelData: Uint8Array,
) {
  const index = (Math.floor(x * width) + Math.floor(y)) * 4

  if (!Number.isFinite(pixelData[index])) {
    debugger
    throw new Error("Pixel index does not exist")
  }

  return pixelData[index]
}
