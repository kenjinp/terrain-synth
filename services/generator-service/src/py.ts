import { spawn } from "child_process"

export const runPy = (path: string) =>
  new Promise(function (resolve, reject) {
    const pyprog = spawn("python3", [path])
    pyprog.stdout.on("data", function (data) {
      resolve(data.toString())
    })
    pyprog.stderr.on("data", data => {
      reject(data.toString())
    })
  })
