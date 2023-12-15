import { useTexture } from "@react-three/drei"
import { BoxGeometryProps, Vector3 } from "@react-three/fiber"
import { FC, useRef } from "react"
import { Color, Mesh, RepeatWrapping, SRGBColorSpace } from "three"

export const Ocean: FC<{
  position: Vector3
  size: BoxGeometryProps["args"]
}> = ({ position, size }) => {
  const ref = useRef<Mesh>(null)
  const n1 = useTexture("/ramp2.jpg")
  const n2 = useTexture("/n5.jpg")
  n1.wrapS = n1.wrapT = RepeatWrapping
  n2.wrapS = n2.wrapT = RepeatWrapping
  n1.colorSpace = SRGBColorSpace
  n1.needsUpdate = true

  return (
    <mesh position={position} receiveShadow>
      <boxGeometry args={size} />
      <meshPhysicalMaterial
        color={new Color(0x4c5a97)}
        sheen={1}
        transparent
        opacity={0.8}
        // normalMap={n2}
      />
      {/* <Water
        mesh={ref.current}
        normalScale={[1, 1]}
        roughness={1.0}
        normalMap={n1}
        surfaceNormal1={n1}
        surfaceNormal2={n2}
        envMapIntensity={1.0}
        transparent
        needsUpdate
        color="#01003b"
      /> */}
    </mesh>
  )
}
