import { useFrame, useThree } from "@react-three/fiber"
import { useEffect, useRef, useState } from "react"
import {
  Box3,
  BoxGeometry,
  DirectionalLight,
  Group,
  Mesh,
  MeshBasicMaterial,
  Vector3,
} from "three"
import { randFloat, randInt } from "three/src/math/MathUtils"
import { Cloud, CloudAnimationState, CloudProps, Clouds } from "./Cloud"

const getRandomPointInBox = (box: Box3): Vector3 => {
  const x = randFloat(box.min.x, box.max.x)
  const y = randFloat(box.min.y, box.max.y)
  const z = randFloat(box.min.z, box.max.z)

  return new Vector3(x, y, z)
}

const tempBoxSize = new Vector3()

export const CloudScroller: React.FC<{
  bounds: Vector3
  position: Vector3
}> = ({ bounds, position }) => {
  const cloudRef = useRef<THREE.Group>(null)
  const [cloudsMap, setCloudsMap] = useState<CloudProps[]>([])
  const scene = useThree(state => state.scene)
  const [wind] = useState(
    new Vector3().random().multiply(new Vector3(1, 0, 1).normalize()),
  )
  const [windSpeed] = useState(randFloat(1, 2))
  const [boundingMesh] = useState(() => {
    const mesh = new Mesh()
    const geo = new BoxGeometry(bounds.x, bounds.y, bounds.z)
    geo.computeBoundingBox()
    mesh.geometry = geo
    mesh.material = new MeshBasicMaterial({
      color: "red",
      wireframe: true,
    })
    return mesh
  })
  const [box3] = useState(new Box3().setFromObject(boundingMesh))
  const [size] = useState(() => box3.getSize(tempBoxSize))

  const createCloud = (v: Vector3) => {
    const scale = Math.random() * 20
    const bounds = new Vector3().random().multiplyScalar(randInt(1, 40))
    const windBlownVector = new Vector3()
      .copy(wind)
      .normalize()
      .multiplyScalar(windSpeed * randInt(1, 50))
    bounds.add(windBlownVector)
    const seed = Math.random()
    const segments = Math.floor(Math.random() * 20 + 10)
    const volume = Math.random() * 100 + 10
    const growth = randFloat(0.1, 0.9)
    const speed = randFloat(0.01, 0.2)
    return {
      id: `${v.x}-${v.y}-${v.z}`,
      seed,
      segments,
      bounds,
      concentrate: "random",
      scale: [scale, scale, scale],
      volume,
      growth,
      speed,
      opacity: randFloat(0.5, 1),
      fade: 0.01,
      position: new Vector3().copy(v),
      state: CloudAnimationState.CHILLING,
    }
  }

  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if (event.key === "t") {
        console.log(cloudRef.current)
      }
    }
    document.addEventListener("keydown", listener)
    return () => document.removeEventListener("keydown", listener)
  }, [])

  // On init, let's create some clouds
  useEffect(() => {
    const clouds: CloudProps[] = []
    const numPoints = randInt(3, 20)
    const points = new Array(numPoints)
      .fill(0)
      .map(() => getRandomPointInBox(box3))
    points.forEach(v => {
      clouds.push(createCloud(v))
    })
    setCloudsMap(clouds)
  }, [box3, wind, windSpeed])

  useFrame((state, delta) => {
    const t = state.clock.getElapsedTime()
    const markedForDeletion: string[] = []
    const markedForCreation: CloudProps[] = []

    if (!cloudRef.current) return
    cloudRef.current.children.forEach((mesh, index) => {
      // all move in the same direction
      if (mesh instanceof Group) {
        mesh.position.addScaledVector(wind, windSpeed)
      }
      // const matchingConfig = cloudsMap.find(({ id }) => id === mesh.name)
      // if (!matchingConfig) {
      //   return
      // }

      // if (matchingConfig.animationState === CloudAnimationState.CREATING) {
      //   const createTime = mesh.userData["creating"] || now
      //   const d = now - createTime

      //   if (mesh.userData["opacity"] !== matchingConfig.opacity) {
      //     mesh.userData["opacity"] =
      //       (mesh.userData["opacity"] || 0) + d * 0.00025
      //   } else if (box3.containsPoint(mesh.position)) {
      //     matchingConfig.animationState = CloudAnimationState.CHILLING
      //   }
      //   mesh.userData["creating"] = now
      // }

      // // delete stuff
      // const deleteTime = mesh.userData["deleting"]
      // // we should decrement opacity
      // if (deleteTime) {
      //   const d = now - deleteTime
      //   mesh.userData["opacity"] =
      //     (mesh.userData["opacity"] || matchingConfig.opacity) - d * 0.00025

      //   if (mesh.userData["opacity"] < 0) {
      //     mesh.visible = false
      //     markedForDeletion.push(matchingConfig.id)
      //     const pos = getRandomPointInBox(box3)
      //     pos.addScaledVector(wind.clone().negate(), size.x + size.x * 0.1)
      //   }
      // }

      // if (matchingConfig.animationState === CloudAnimationState.CREATING) {
      // } else if (!box3.containsPoint(mesh.position)) {
      //   mesh.userData["deleting"] = Date.now()
      // }
      // }
    })
    // if (markedForCreation.length > 0 || markedForDeletion.length > 0) {
    //   setCloudsMap([
    //     ...cloudsMap.filter(({ id }) => !markedForDeletion.includes(id)),
    //     ...markedForCreation,
    //   ])
    // }
  })

  scene.children.forEach(child => {
    if (child instanceof DirectionalLight && !child.name.includes("sun")) {
      ;(child as DirectionalLight).intensity = 0
    }
  })

  const dirLightPosition =
    scene.children.find(
      child => child instanceof DirectionalLight && child.name.includes("sun"),
    )?.position || new Vector3()

  return (
    <mesh position={position}>
      {/* <box3Helper args={[box3, "yellow"]} /> */}
      <Clouds
        ref={cloudRef}
        frustumCulled={false}
        castShadow
        limit={10_000}
        lightOrigin={dirLightPosition}
      >
        {cloudsMap.map(cloudProps => (
          <Cloud key={cloudProps.id} {...cloudProps} fade={0} />
        ))}
      </Clouds>
    </mesh>
  )
}
