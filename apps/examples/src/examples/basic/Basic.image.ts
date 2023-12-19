export function createImageElementFromImageData(
  imageData: ImageData,
): HTMLImageElement {
  // Create a canvas element to draw the ImageData
  const canvas = document.createElement("canvas")
  const context = canvas.getContext("2d")

  if (!context) {
    throw new Error("Canvas 2D context is not supported.")
  }

  // Set the canvas size to match the ImageData size
  canvas.width = imageData.width
  canvas.height = imageData.height

  // Put the ImageData onto the canvas
  context.putImageData(imageData, 0, 0)

  // Create an HTML image element
  const imageElement = new Image()

  // Set the image source to the canvas data URL
  imageElement.src = canvas.toDataURL()
  imageElement.style.zIndex = "9999"
  // imageElement.id = "ocean heights"
  // document.body.appendChild(imageElement)

  return imageElement
}

export function processImageData(inputImageData: ImageData): ImageData {
  const width = inputImageData.width
  const height = inputImageData.height

  // Create a copy of the input ImageData to store the result
  const outputImageData = new ImageData(width, height)

  // Helper function to calculate the distance between two points
  const calculateDistance = (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
  ): number => {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))
  }

  // Iterate through each pixel
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4 // Each pixel has 4 values: red, green, blue, and alpha

      // Check if the pixel value is 0
      if (inputImageData.data[index] === 0) {
        let minDistance = Number.MAX_VALUE

        // Iterate through each pixel to find the minimum distance to a non-zero pixel
        for (let ny = 0; ny < height; ny++) {
          for (let nx = 0; nx < width; nx++) {
            const nIndex = (ny * width + nx) * 4

            // Check if the pixel value is greater than 0
            if (inputImageData.data[nIndex] > 0) {
              const distance = calculateDistance(x, y, nx, ny)

              // Update minDistance if the current distance is smaller
              minDistance = Math.min(minDistance, distance)
            }
          }
        }

        // Set the pixel value in the output ImageData based on the minimum distance
        outputImageData.data[index] = minDistance
        outputImageData.data[index + 1] = minDistance
        outputImageData.data[index + 2] = minDistance
        outputImageData.data[index + 3] = 255 // Set alpha channel to fully opaque
      } else {
        // // If the pixel value is already greater than 0, copy the value to the output ImageData
        // outputImageData.data.set(
        //   inputImageData.data.subarray(index, index + 4),
        //   index,
        // )
        outputImageData.data[index] = 0
        outputImageData.data[index + 1] = 0
        outputImageData.data[index + 2] = 0
        outputImageData.data[index + 3] = 255
      }
    }
  }

  return outputImageData
}
