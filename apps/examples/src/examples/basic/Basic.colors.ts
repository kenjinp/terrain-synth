import { Noise } from "@hello-worlds/planets"

export function interpolateColor(
  x: number,
  y: number,
  width: number,
  pixelData: Uint8Array,
  noise: Noise,
) {
  let n = 1 - noise.get(x + 500, 0, y + 500)
  let e = noise.get(x + 500 + n, 0, y + n + 500)
  x = x + n + e
  y = y + n + e
  // Get the integer and fractional parts of the coordinates
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  const x1 = x0 + 1
  const y1 = y0 + 1

  // Ensure the coordinates are within the image bounds
  // if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height) {
  //   throw new Error('Interpolation coordinates are outside the image bounds');
  // }

  // Get the colors of the four surrounding pixels
  const topLeft = getPixelColor(x0, y0, width, pixelData)
  const topRight = getPixelColor(x1, y0, width, pixelData)
  const bottomLeft = getPixelColor(x0, y1, width, pixelData)
  const bottomRight = getPixelColor(x1, y1, width, pixelData)
  // const n = 1
  // Calculate the weights based on the fractional parts
  const weightX = x - x0
  const weightY = y - y0

  // const biasX1 = remap(weightX, 0, 1, 1, 0)
  // const biasY1 = remap(weightY, 0, 1, 1, 0)
  // const biasX1 = remap(weightX, 0, 0.5, 1, 0)
  // const biasY1 = remap(weightY, 0, 0.5, 1, 0)
  // n * (biasX + biasY)

  // Interpolate the color values
  // const interpolatedColor = [
  //   interpolateChannel(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0], weightX, weightY),
  //   interpolateChannel(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1], weightX, weightY),
  //   interpolateChannel(topLeft[2], topRight[2], bottomLeft[2], bottomRight[2], weightX, weightY),
  // ];

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

function getPixelColor(
  x: number,
  y: number,
  width: number,
  pixelData: Uint8Array,
) {
  // const index = (Math.floor(y) * imageData.width + Math.floor(x)) * 4
  // // return [
  // //   imageData.data[index],
  // //   // imageData.data[index + 1],
  // //   // imageData.data[index + 2],
  // // ]
  // return imageData.data[index]
  const halfWidth = width / 2

  // get the 3 closest pixels

  // Convert position to flat array index
  const index =
    (Math.floor(x + halfWidth) * width + Math.floor(y + halfWidth)) * 4

  return pixelData[index] ? pixelData[index] : 0
}
