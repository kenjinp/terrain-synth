// Load model
// attempt to run model
// if fails, fallback to wasm

import * as onnxruntimeWeb from "onnxruntime-web"
import { loadModel, runModel } from "./Model.gan"

let model: onnxruntimeWeb.InferenceSession

onmessage = async (event: MessageEvent) => {
  const { command, seed } = event.data
  console.log({ command, seed })
  try {
    if (command === "load") {
      const { model: m } = await loadModel(postMessage, seed)
      model = m
    } else if (command === "run") {
      await runModel(model, postMessage, seed)
    } else {
      throw new Error("Unknown command: " + command)
    }
  } catch (err) {
    if (err instanceof Error) {
      postMessage({ error: err.message })
    }
  }
}
