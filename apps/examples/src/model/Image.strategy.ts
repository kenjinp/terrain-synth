import { sample } from "@hello-worlds/core"
import img1 from "../../../../data/elevations/class_1/0001_elevation.png?url"
import img2 from "../../../../data/elevations/class_1/0002_elevation.png?url"
import img3 from "../../../../data/elevations/class_1/0003_elevation.png?url"
import img4 from "../../../../data/elevations/class_1/0005_elevation.png?url"
// import img2 from "../assets/wgan-gp-test-2.png?url"
// import img3 from "../assets/wgan-gp-test-3.png?url"
// import img4 from "../assets/wgan-gp-test-4.png?url"
import { MODEL_STATE } from "./Model.gan"
import { getImageDataFromImg } from "./image"

const images = [img1, img2, img3, img4]

export class ImageStrategy {
  state: MODEL_STATE = MODEL_STATE.IDLE
  listeners: ((state: MODEL_STATE) => void)[] = []
  constructor() {}

  addStateListener(listener: (state: MODEL_STATE) => void) {
    this.listeners.push(listener)
  }

  load(): Promise<ImageData> {
    return new Promise((resolve, _) => {
      this.listeners.forEach(l => l(MODEL_STATE.LOADING))
      const src = sample(images) as unknown as string

      const image = document.createElement("img")
      image.src = src // image.src = "IMAGE URL/PATH
      console.log(image, src)
      image.onload = () => {
        this.listeners.forEach(l => l(MODEL_STATE.IDLE))
        resolve(getImageDataFromImg(image))
      }
    })
  }

  async run() {
    return await this.load()
  }
}
