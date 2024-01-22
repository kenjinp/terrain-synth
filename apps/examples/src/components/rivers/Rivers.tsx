import React from "react"
import { Color, InstancedMesh, Matrix4, Quaternion, Vector3 } from "three"
import { River } from "./River.model"

const parentMatrix = /* @__PURE__ */ new Matrix4()
const translation = /* @__PURE__ */ new Vector3()
const rotation = /* @__PURE__ */ new Quaternion()
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

export const RiverContext = React.createContext<River>(null!)

export const useRiver = () => {
  return React.useContext(RiverContext)
}

export const RiverNode: React.FC = () => {
  const river = useRiver()

  return null
}

export const Rivers: React.FC = () => {
  const number = 64
  const size = 10_000
  const jitter = 10
  const limit = number ** 2

  const instance = React.useRef<InstancedMesh>(null!)

  const [river] = React.useState(() => {
    return new River(number, size, jitter)
  })

  React.useLayoutEffect(() => {
    const positions = river.delaunay.points
    const count = river.delaunay.points.length / 2
    instance.current.count = count

    console.log({ count })

    setUpdateRange(instance.current.instanceMatrix, {
      offset: 0,
      count: count * 16,
    })

    parentMatrix.copy(instance.current.matrixWorld).invert()

    const color = new Color()
    for (let index = 0; index < positions.length; index += 2) {
      const x = positions[index]
      const y = positions[index + 1]
      const z = 1000 //river.heightmap.get(x, z)

      if (y > 0) {
        console.log("omg")
      }

      translation.set(x, y, z)
      rotation.set(0, 0, 0, 1)
      scale.set(1, 1, 1)
      tempMatrix.compose(translation, rotation, scale).premultiply(parentMatrix)
      instance.current.setMatrixAt(index, tempMatrix)
      instance.current.setColorAt(index, color.set(Math.random() * 0xffffff))
    }
    console.log({ instance: instance.current })
    instance.current.instanceMatrix.needsUpdate = true
  }, [river])

  return (
    <RiverContext.Provider value={river}>
      <axesHelper args={[10000]} position-y={1000} />
      <instancedMesh
        ref={instance}
        matrixAutoUpdate={false}
        args={[null as any, null as any, limit]}
      >
        <sphereGeometry args={[10, 8, 8]} />
        <meshStandardMaterial />
      </instancedMesh>
    </RiverContext.Provider>
  )
}
