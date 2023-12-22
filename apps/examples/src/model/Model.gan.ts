import * as onnxruntimeWeb from "onnxruntime-web"
import generator from "../assets/generator.onnx?url"
import { createImageDataFromArray } from "./image"
import { generateStandardNormalArray } from "./math"

// This is for the WGAN model
const modelImageOutputSize = 128
const batchSize = 1
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

function generateFeeds(model: onnxruntimeWeb.InferenceSession, seed: string) {
  const randomArray = generateStandardNormalArray(array, size, seed)
  const buffer = new Float32Array(randomArray)
  const inputTensor = new onnxruntimeWeb.Tensor("float32", buffer, dims)

  const feeds: Record<string, onnxruntimeWeb.Tensor> = {}
  feeds[model.inputNames[0]] = inputTensor

  return feeds
}

export async function runModel(
  model: onnxruntimeWeb.InferenceSession,
  postMessage: (message: any) => void,
  seed = Math.random().toString(36).substring(6),
) {
  postMessage({ state: MODEL_STATE.RUNNING })
  console.time("inference")
  const feeds = generateFeeds(model, seed)
  const out = await model.run(feeds)
  console.timeEnd("inference")
  console.time("image")
  const outputBuffer = out[model.outputNames[0]].data as Float32Array
  const result = createImageDataFromArray(modelImageOutputSize, outputBuffer)
  console.timeEnd("image")
  postMessage({ state: MODEL_STATE.IDLE })
  postMessage({ result })
  return result
}
let model: onnxruntimeWeb.InferenceSession

export async function loadModel(
  postMessage: (message: any) => void,
  seed?: string,
) {
  let result: ImageData
  // first try to load webgl
  // and run it
  // if it fails, fallback to wasm
  postMessage({ state: MODEL_STATE.LOADING })

  if (model) {
    // model already loaded
    // just run it
    result = await runModel(model, postMessage, seed)
    return { model, result }
  }

  try {
    model = await onnxruntimeWeb.InferenceSession.create(generator, {
      executionProviders: [strategy[0]],
    })
    postMessage({ state: MODEL_STATE.LOADED })
    result = await runModel(model, postMessage, seed)
  } catch (e) {
    try {
      console.error("webgl failed", e)
      model = await onnxruntimeWeb.InferenceSession.create(generator, {
        executionProviders: [strategy[1]],
      })
      postMessage({ state: MODEL_STATE.LOADED })
      result = await runModel(model, postMessage, seed)
    } catch (e) {
      console.error("wasm failed", e)
      throw new Error("Failed to load model")
    }
  }
  return { model, result }
}
