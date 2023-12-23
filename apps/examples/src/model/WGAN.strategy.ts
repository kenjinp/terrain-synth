import "../assets/ort-wasm-simd-threaded.wasm?url"
import { MODEL_STATE } from "./Model.gan"
import ModelWorker from "./Model.worker?worker"

export class WGANStrategy {
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

  command<
    T = {
      terrainData: Uint8Array
      oceanData: Uint8Array
    },
  >(command: string, message?: Record<string, any>) {
    return new Promise<T>((resolve, reject) => {
      this.worker.postMessage({ command, ...message })
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

  load(seed?: string) {
    return this.command("load", { seed })
  }

  run(seed?: string) {
    return this.command("run", { seed })
  }
}
