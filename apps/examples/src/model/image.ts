import { remap } from "@hello-worlds/planets"

// Function to crop the offscreen canvas
function cropCanvas(
  originalCanvas: OffscreenCanvas,
  leftCrop: number,
  topCrop: number,
  rightCrop: number,
  bottomCrop: number,
): OffscreenCanvas {
  // Calculate the new dimensions
  const newWidth: number = originalCanvas.width - leftCrop - rightCrop
  const newHeight: number = originalCanvas.height - topCrop - bottomCrop

  // Create a new offscreen canvas with the new dimensions
  const croppedCanvas = new OffscreenCanvas(newWidth, newHeight)

  // Get the 2D context of the new canvas
  const croppedContext = croppedCanvas.getContext("2d")!

  // Draw the cropped region onto the new canvas
  croppedContext.drawImage(
    originalCanvas,
    leftCrop, // X-coordinate to start cropping
    topCrop, // Y-coordinate to start cropping
    newWidth, // Width of the cropped region
    newHeight, // Height of the cropped region
    0, // X-coordinate to place the cropped region on the new canvas
    0, // Y-coordinate to place the cropped region on the new canvas
    newWidth, // Width of the drawn image on the new canvas
    newHeight, // Height of the drawn image on the new canvas
  )

  // Return the cropped canvas
  return croppedCanvas
}

function resizeCanvas(
  offscreenCanvas: OffscreenCanvas,
  width: number,
  height: number,
): OffscreenCanvas {
  // Create a new offscreen canvas with the new dimensions
  const resizedCanvas = new OffscreenCanvas(width, height)

  // Get the 2D context of the new canvas
  const resizedContext = resizedCanvas.getContext("2d")!

  // Draw the resized canvas
  resizedContext.drawImage(offscreenCanvas, 0, 0, width, height)

  return resizedCanvas
}

function blurCanvas(
  offscreenCanvas: OffscreenCanvas,
  pixels: number,
): OffscreenCanvas {
  // Create a new offscreen canvas with the same dimensions
  const blurredCanvas = new OffscreenCanvas(
    offscreenCanvas.width,
    offscreenCanvas.height,
  )

  // Get the 2D context of the new canvas
  const blurredContext = blurredCanvas.getContext("2d")!

  // blur canvas
  blurredContext.filter = `blur(${pixels}px)`

  // Draw the blurred canvas
  blurredContext.drawImage(offscreenCanvas, 0, 0)

  return blurredCanvas
}

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

  const blurRadius = 0

  // Draw the image data on the canvas
  context!.putImageData(imageData, 0, 0)

  const blurredCanvas = cropCanvas(blurCanvas(canvas, blurRadius), 1, 1, 1, 1)
  // Get the 2D rendering context
  const blurredContext = blurredCanvas.getContext("2d")!

  console.log({ blurredCanvas })

  return blurredContext.getImageData(
    0,
    0,
    blurredCanvas.width,
    blurredCanvas.height,
  )
}

export const getImageDataFromImg = (img: HTMLImageElement): ImageData => {
  const canvas = document.createElement("canvas")
  canvas.width = img.width
  canvas.height = img.height
  const context = canvas.getContext("2d")!
  context.drawImage(img, 0, 0)
  return context.getImageData(0, 0, img.width, img.height)
}

export const convertImageDataToShardArrayBuffer = (
  imageData: ImageData,
): Uint8Array => {
  const { data } = imageData
  const result = new Uint8Array(new SharedArrayBuffer(data.length))
  for (let i = 0; i < data.length; i++) {
    result[i] = data[i]
  }
  return result
}
