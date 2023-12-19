import * as onnxruntimeWeb from "onnxruntime-web"
import generator from "../../../../.terrain-ProGAN/generator.onnx?url"
import "../assets/ort-wasm-simd-threaded.wasm?url"
import { createImageDataFromArray } from "./image"
import { generateStandardNormalArray } from "./math"

const modelImageOutputSize = 128
const batchSize = 8
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
  console.log({ model })
  const feeds: Record<string, onnxruntimeWeb.Tensor> = {}
  feeds[model.inputNames[0]] = inputTensor
  // feeds[model.inputNames[1]] = new onnxruntimeWeb.Tensor("int32", [], [0])
  // feeds[model.inputNames[2]] = new onnxruntimeWeb.Tensor("int8", [3])

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
