import * as onnxruntimeWeb from "onnxruntime-web"
import "../assets/ort-wasm-simd-threaded.wasm?url"
import { MODEL_STATE, loadModel, runModel } from "./Model.utils"
import ModelWorker from "./Model.worker?worker"

export class ThreadedTerrainSynthModel {
  worker = new ModelWorker()
  state: MODEL_STATE = MODEL_STATE.IDLE
  constructor() {}

  addStateListener(listener: (state: MODEL_STATE) => void) {
    this.worker.addEventListener("message", event => {
      const { state, result, error } = event.data
      if (error) {
        console.error(error)
      }
      if (state) {
        this.state = state
      }
      if (result) {
        this.state = MODEL_STATE.IDLE
      }
      listener(this.state)
    })
  }

  command<T = ImageData>(command: string) {
    return new Promise<T>((resolve, reject) => {
      this.worker.postMessage({ command })
      this.worker.onmessage = event => {
        const { result, error } = event.data
        if (error) {
          return reject(error)
        }
        if (result) {
          return resolve(result)
        }
      }
    })
  }

  load() {
    return this.command("load")
  }

  run() {
    return this.command("run")
  }
}

export class TerrainSynthModel {
  state: MODEL_STATE = MODEL_STATE.IDLE
  model?: onnxruntimeWeb.InferenceSession
  listeners: ((state: MODEL_STATE) => void)[] = []
  constructor() {}

  addStateListener(listener: (state: MODEL_STATE) => void) {
    this.listeners.push((event: any) => {
      const { state, result, error } = event
      if (error) {
        console.error(error)
      }
      if (state) {
        this.state = state
      }
      if (result) {
        this.state = MODEL_STATE.IDLE
      }
      listener(this.state)
    })
  }

  postMessage(message: any) {
    this.listeners.forEach(listener => listener(message))
  }

  async load() {
    const { model, result } = await loadModel(this.postMessage.bind(this))
    this.model = model
    return result
  }

  async run() {
    if (!this.model) {
      throw new Error("Model not loaded")
    }
    return await runModel(this.model!, this.postMessage.bind(this))
  }
}
