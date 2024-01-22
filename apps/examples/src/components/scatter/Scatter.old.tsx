import { FC, useLayoutEffect, useRef } from "react"
import { InstancedMesh, Matrix4, Quaternion, Vector3 } from "three"

const parentMatrix = /* @__PURE__ */ new Matrix4()
const translation = /* @__PURE__ */ new Vector3()
const rotation = /* @__PURE__ */ new Quaternion()
const cpos = /* @__PURE__ */ new Vector3()
const cquat = /* @__PURE__ */ new Quaternion()
const scale = /* @__PURE__ */ new Vector3()
const tempMatrix = /* @__PURE__ */ new Matrix4()

const setUpdateRange = (
  attribute: THREE.BufferAttribute,
  updateRange: { offset: number; count: number },
): void => {
  if ("updateRanges" in attribute) {
    // r159
    // @ts-ignore
    attribute.updateRanges[0] = updateRange
  } else {
    attribute.updateRange = updateRange
  }
}

export const Scatter: FC<{ positions: number[][] }> = ({ positions }) => {
  const size = 20
  const instance = useRef<InstancedMesh>(null!)
  const limit = 20_000
  // console.log({ nodes, materials })

  // const [positions] = useState(() => {
  //   const tempVec3 = new Vector3()
  //   const noise = new Noise({
  //     seed: "trees",
  //     height: 2,
  //     scale: 1000,
  //   })

  //   const p = new PoissonDiskSampling({
  //     shape: [10000, 10000],
  //     minDistance: 40,
  //     maxDistance: 500,
  //     tries: 10,
  //     distanceFunction: function (point) {
  //       return noise.get(point[0], 0, point[1])
  //     },
  //   })
  //   let points = p.fill()
  //   // sample max 10k points
  //   points.length = Math.min(points.length, limit)
  //   points = points.slice(0, limit)

  //   for (let index = 0; index < points.length; index++) {
  //     const point = points[index]
  //     tempVec3.set(point[0], 0, point[1]).sub(new Vector3(5000, 0, 5000))
  //     points[index] = tempVec3.toArray()
  //   }
  //   return points
  // })

  useLayoutEffect(() => {
    const count = positions.length
    instance.current.count = count
    setUpdateRange(instance.current.instanceMatrix, {
      offset: 0,
      count: count * 16,
    })

    parentMatrix.copy(instance.current.matrixWorld).invert()

    for (let index = 0; index < positions.length; index++) {
      translation.set(
        positions[index][0],
        positions[index][1],
        positions[index][2],
      )
      rotation.set(0, 0, 0, 1)
      scale.set(1, 1, 1)
      tempMatrix.compose(translation, rotation, scale).premultiply(parentMatrix)
      instance.current.setMatrixAt(index, tempMatrix)
    }
    instance.current.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh
      ref={instance}
      matrixAutoUpdate={false}
      // frustumCulled={false}
      args={[null as any, null as any, limit]}
      // geometry={nodes.tree01_top.geometry}
      // material={nodes.tree01_top.material}
      // castShadow
    >
      <coneGeometry args={[Math.floor(size / 4), size, 8]} />
      <meshStandardMaterial color="green" />
    </instancedMesh>
  )
}
