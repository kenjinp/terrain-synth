import * as onnxruntimeWeb from "onnxruntime-web"
import generator from "../assets/generator.onnx?url"
import "../assets/ort-wasm-simd-threaded.wasm?url"
import ModelWorker from "./Model.worker?worker"
import { createImageDataFromArray } from "./image"
import { generateStandardNormalArray } from "./math"

const modelImageOutputSize = 64
const batchSize = 64
const latentDim = 100
const dims = [batchSize, latentDim, 1, 1]
const size = dims.reduce((a, b) => a * b)
const array = new Array(size).fill(0)
const strategy = ["webgl", "wasm"]

export const hyperparameters = {
  modelImageOutputSize,
  batchSize,
  latentDim,
  dims,
  size,
  array,
}

export enum MODEL_STATE {
  LOADING,
  LOADED,
  RUNNING,
  IDLE,
}

function generateFeeds(model: onnxruntimeWeb.InferenceSession) {
  const randomArray = generateStandardNormalArray(array, size)
  const buffer = new Float32Array(randomArray)
  const inputTensor = new onnxruntimeWeb.Tensor("float32", buffer, dims)

  const feeds: Record<string, onnxruntimeWeb.Tensor> = {}
  feeds[model.inputNames[0]] = inputTensor

  return feeds
}

export async function runModel(
  model: onnxruntimeWeb.InferenceSession,
  postMessage: (message: any) => void,
) {
  postMessage({ state: MODEL_STATE.RUNNING })
  console.time("inference")
  const feeds = generateFeeds(model)
  const out = await model.run(feeds)
  console.timeEnd("inference")
  console.time("image")
  const outputBuffer = out[model.outputNames[0]].data as Float32Array
  console.log({ outputBuffer })
  const result = createImageDataFromArray(modelImageOutputSize, outputBuffer)
  console.timeEnd("image")
  postMessage({ result })
  return result
}

export async function loadModel(postMessage: (message: any) => void) {
  let model: onnxruntimeWeb.InferenceSession
  let result: ImageData
  // first try to load webgl
  // and run it
  // if it fails, fallback to wasm
  postMessage({ state: MODEL_STATE.LOADING })
  try {
    model = await onnxruntimeWeb.InferenceSession.create(generator, {
      executionProviders: [strategy[0]],
    })
    postMessage({ state: MODEL_STATE.LOADED })
    result = await runModel(model, postMessage)
  } catch (e) {
    try {
      console.error("webgl failed", e)
      model = await onnxruntimeWeb.InferenceSession.create(generator, {
        executionProviders: [strategy[1]],
      })
      postMessage({ state: MODEL_STATE.LOADED })
      result = await runModel(model, postMessage)
    } catch (e) {
      console.error("wasm failed", e)
      throw new Error("Failed to load model")
    }
  }
  return { model, result }
}

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
