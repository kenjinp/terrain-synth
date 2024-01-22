import { BoxGeometryProps, type Vector3 } from "@react-three/fiber"
import { FC, useRef } from "react"
import { useSun } from "../../hooks/use-sun"
import { WaterMaterial } from "./Water.material"

export const Ocean: FC<{
  position: Vector3
  size: BoxGeometryProps["args"]
}> = ({ position = [0, 0, 0], size = [10, 1, 10] }) => {
  const light = useSun()
  const ref = useRef<THREE.Mesh>(null)

  return light ? (
    <mesh ref={ref} position={position} receiveShadow>
      <boxGeometry args={size} />
      {ref.current && (
        <WaterMaterial sunPosition={light.position} mesh={ref.current} />
      )}
    </mesh>
  ) : null
}
