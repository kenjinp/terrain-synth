import { useFrame, useThree } from "@react-three/fiber"
import { useState } from "react"

export function useSun() {
  const [_, setBlah] = useState(0)
  const light = useThree(state => {
    const light = state.scene.getObjectByName(
      "sun-shadow",
    ) as THREE.DirectionalLight
    return light
  })

  useFrame(() => {
    if (light?.shadow?.map?.texture) {
      setBlah(1)
    }
  })

  return light
}
