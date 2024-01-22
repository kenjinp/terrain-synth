import { useGLTF, useTexture } from "@react-three/drei"
import { useLayoutEffect } from "react"
import tree01 from "./tree01.glb?url"
import treeSpring from "./tree01_spring.png?url"

export function Tree(props) {
  const { nodes, materials } = useGLTF(tree01)
  const [mat] = useTexture([treeSpring])
  useLayoutEffect(() => {
    if (nodes.tree01_top.material) {
      nodes.tree01_top.geometry.computeBoundingSphere()
      nodes.tree01_top.material.map = mat
    }
    console.log(nodes.tree01_top.geometry)
  })
  return (
    <mesh>
      <boxGeometry args={[20, 20, 20]} />
      {/* <primitive object={nodes.tree01_top} /> */}
      <meshStandardMaterial color="green" />
      {/* <mesh
          castShadow
          receiveShadow
          geometry={nodes.tree01_top.geometry}
          material={nodes.tree01_top.material}
          position={[0, 2.4, 0.001]}
        /> */}
    </mesh>
  )
}

useGLTF.preload(tree01)
