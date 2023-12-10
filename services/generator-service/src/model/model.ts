import * as ort from "onnxruntime-node"
const modelPath = `./src/model/model.onnx`

export const loadModel = async () => {
  const session = await ort.InferenceSession.create(modelPath)
  return async function runModel() {
    // prepare inputs. a tensor need its corresponding TypedArray as data
    const dataA = Float32Array.from(
      new Array(32 * 32).fill(0).map(() => Math.random()),
    )
    const arg0 = new ort.Tensor("float32", dataA, [1, 1, 32, 32])

    // prepare feeds. use model input names as keys.
    const feeds = { arg0 }
    // feed inputs and run
    const results = await session.run(feeds)

    return results
  }
}
