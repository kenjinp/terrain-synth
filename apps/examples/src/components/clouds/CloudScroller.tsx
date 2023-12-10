import { Cloud, Clouds } from "@react-three/drei"
import { ReactThreeFiber, useFrame } from "@react-three/fiber"
import { useEffect, useRef, useState } from "react"
import { Box3, BoxGeometry, Mesh, MeshBasicMaterial, Vector3 } from "three"
import { randFloat, randInt } from "three/src/math/MathUtils"

interface CloudProps {
  key: string
  seed?: number
  segments?: number
  bounds?: ReactThreeFiber.Vector3
  concentrate?: "random" | "inside" | "outside"
  scale?: ReactThreeFiber.Vector3
  volume?: number
  smallestVolume?: number
  growth?: number
  speed?: number
  fade?: number
  opacity?: number
  color?: ReactThreeFiber.Color
  position?: ReactThreeFiber.Vector3
}

const getRandomPointInBox = (box: Box3): Vector3 => {
  const x = randFloat(box.min.x, box.max.x)
  const y = randFloat(box.min.y, box.max.y)
  const z = randFloat(box.min.z, box.max.z)

  return new Vector3(x, y, z)
}

export const CloudScroller: React.FC<{
  bounds: Vector3
  position: Vector3
}> = ({ bounds, position }) => {
  const cloudRef = useRef<THREE.Group>(null)
  const [cloudsMap, setCloudsMap] = useState<CloudProps[]>([])
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

  // On init, let's create some clouds
  useEffect(() => {
    const clouds: CloudProps[] = []
    const numPoints = randInt(3, 20)
    const points = new Array(numPoints)
      .fill(0)
      .map(() => getRandomPointInBox(box3))
    points.forEach(v => {
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
      clouds.push({
        key: `${v.x}-${v.y}-${v.z}`,
        seed,
        segments,
        bounds,
        concentrate: "random",
        scale: [scale, scale, scale],
        volume,
        growth,
        speed,
        opacity: randFloat(0.5, 1),
        fade: -Infinity,
        position: new Vector3().copy(v),
      })
      setCloudsMap(clouds)
    })
  }, [box3, wind, windSpeed])

  useFrame(() => {
    if (!cloudRef.current) return
    // console.log(clouds.current)
    cloudRef.current.children.forEach(cloud => {
      cloud.position.addScaledVector(wind, windSpeed)

      // if cloud goes beyond bounds, reset it
      if (!box3.containsPoint(cloud.position)) {
        cloud.position.copy(getRandomPointInBox(box3))
      }
    })
  })

  return (
    <mesh position={position}>
      {/* <primitive object={boundingMesh} /> */}
      {/* <box3Helper args={[box3, new Color("yellow")]} /> */}
      <Clouds ref={cloudRef} frustumCulled={false}>
        {cloudsMap.map(({ key, ...cloudProps }) => (
          <Cloud key={key} {...cloudProps} castShadow />
        ))}
      </Clouds>
    </mesh>
  )
}
