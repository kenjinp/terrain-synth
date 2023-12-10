import { BoxGeometryProps, Vector3 } from "@react-three/fiber"
import { FC } from "react"
import { Color } from "three"

export const Ocean: FC<{
  position: Vector3
  size: BoxGeometryProps["args"]
}> = ({ position, size }) => {
  return (
    <mesh position={position}>
      <boxGeometry args={size} />
      <meshPhysicalMaterial color={new Color(0x4c5a97)} sheen={1} />
    </mesh>
  )
}
