import { randomRange } from "@hello-worlds/core"
import { Delaunay } from "d3-delaunay"

const createJitteredGrid = (
  gridNumber: number,
  gridSize: number,
  jitter: number,
) => {
  const absJitter = Math.abs(jitter)
  const offsets: number[] = []
  let minX = 0
  let maxX = 0
  let minY = 0
  let maxY = 0
  for (let i = 0; i < gridNumber; i++) {
    const x = i / gridNumber - 0.5
    for (let j = 0; j < gridNumber; j++) {
      const y = j / gridNumber - 0.5
      const xVal =
        x * gridSize + (jitter > 0 ? randomRange(-absJitter, absJitter) : 0)
      const yVal =
        y * gridSize + (jitter > 0 ? randomRange(-absJitter, absJitter) : 0)

      minX = Math.min(minX, xVal)
      maxX = Math.max(maxX, xVal)
      minY = Math.min(minY, yVal)
      maxY = Math.max(maxY, yVal)
      offsets.push(xVal, yVal)
    }
  }
  console.log({ minX, maxX, minY, maxY })
  return new Float32Array(offsets)
}

export class River {
  public delaunay: Delaunay<any>
  constructor(
    public gridNumber: number,
    public gridSize: number,
    public jitter: number,
  ) {
    // generate grid of size gridSize
    const grid = createJitteredGrid(gridNumber, gridSize, jitter)

    // generate height for each position in grid

    // sort grid by height

    // recursively ->

    // find neighbors
    // get slope for each neighbor
    // find steepest slope downwards
    // move to that neighbor
    // add current node index id to 'source' of that neighbor
    // repeat until no neighbor is 'lower'
    // move to next heighest node
    // repeat until all nodes are visited

    this.delaunay = new Delaunay(grid)
    console.log(grid.length, this.delaunay)
  }
}
