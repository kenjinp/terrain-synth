import express from "express"
import { dirname } from "path"
import stream from "stream"
import { fileURLToPath } from "url"
import { getRandomImage, readImage } from "./examples.ts"
import { loadModel } from "./model/model.ts"
import { runPy } from "./py.ts"
const __filenameNew = fileURLToPath(import.meta.url)

const __dirnameNew = dirname(__filenameNew)

const app = express()
const port = 3000
;(async function main() {
  const forwardModel = await loadModel()

  app.get("/", (req, res) => {
    res.send("Hello World!")
  })

  app.get("/random", async (req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*")
    res.setHeader("Access-Control-Allow-Methods", "*")
    res.setHeader("Access-Control-Allow-Headers", "*")
    const r = await getRandomImage("../../../data/elevations/class_1")
    const ps = new stream.PassThrough() // <---- this makes a trick with stream error handling

    stream.pipeline(
      r,
      ps, // <---- this makes a trick with stream error handling
      err => {
        if (err) {
          console.log(err) // No such file or any other kind of error
          return res.sendStatus(400)
        }
      },
    )
    ps.pipe(res)
  })

  app.get("/generate", async (req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*")
    res.setHeader("Access-Control-Allow-Methods", "*")
    res.setHeader("Access-Control-Allow-Headers", "*")
    try {
      const result = await runPy("src/model/model_worker.py")
      console.log("result", result)
    } catch (error) {
      console.error(error)
    }
    const r = await readImage("../temp.png")
    const ps = new stream.PassThrough() // <---- this makes a trick with stream error handling

    stream.pipeline(
      r,
      ps, // <---- this makes a trick with stream error handling
      err => {
        if (err) {
          console.log(err) // No such file or any other kind of error
          return res.sendStatus(400)
        }
      },
    )
    ps.pipe(res)
  })

  app.listen(port, () => {
    return console.log(`Express is listening at http://localhost:${port}`)
  })
})()
