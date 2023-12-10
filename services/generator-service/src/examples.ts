import * as fs from "fs"
import * as path from "path"
import { __dirname } from "./utils.ts"

export const readImage = (imagePath: string): Promise<fs.ReadStream> => {
  return new Promise((resolve, reject) => {
    const r = fs.createReadStream(path.resolve(__dirname, imagePath))
    resolve(r)
  })
}

export const getRandomImage = (folderPath: string): Promise<fs.ReadStream> => {
  return new Promise((resolve, reject) => {
    // Read all files in the folder
    fs.readdir(path.resolve(__dirname, folderPath), (err, files) => {
      if (err) {
        return reject(err)
      }

      // Filter out non-image files (you can add more extensions if needed)
      const imageFiles = files.filter(file =>
        /\.(jpg|jpeg|png|gif)$/i.test(file),
      )

      if (imageFiles.length === 0) {
        return reject(
          new Error("No image files found in the specified folder."),
        )
      }

      // Choose a random image file
      const randomImageFile =
        imageFiles[Math.floor(Math.random() * imageFiles.length)]
      const imagePath = path.join(folderPath, randomImageFile)
      console.log("PATH", path.resolve(__dirname, imagePath))
      const r = fs.createReadStream(path.resolve(__dirname, imagePath))
      resolve(r)
    })
  })
}
