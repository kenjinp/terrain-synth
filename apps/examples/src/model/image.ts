import { remap } from "@hello-worlds/planets"

export function createImageDataFromArray(
  canvasSize: number,
  dataArray: Float32Array,
): ImageData {
  // Create a canvas element
  const canvas = new OffscreenCanvas(canvasSize, canvasSize)
  canvas.width = canvasSize
  canvas.height = canvasSize

  // Get the 2D rendering context
  const context = canvas.getContext("2d")

  if (!context) {
    throw new Error("Canvas 2D context is not supported")
  }

  // construct an image with 4 channels using the single input channel
  const normalizedData = new Uint8ClampedArray(canvasSize * canvasSize * 4)
  for (let i = 0; i < canvasSize * canvasSize; i++) {
    const value = dataArray[i]
    const normalizedValue = Math.floor(remap(value, -1, 1, 0, 255)) // Math.round((value + 1) * 127.5)
    const index = i * 4
    normalizedData[index] = normalizedValue
    normalizedData[index + 1] = normalizedValue
    normalizedData[index + 2] = normalizedValue
    normalizedData[index + 3] = 255
  }

  // Create ImageData object
  const imageData = new ImageData(normalizedData, canvasSize, canvasSize)

  // Draw the image data on the canvas
  context!.putImageData(imageData, 0, 0)

  return imageData
}
