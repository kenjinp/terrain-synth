import { useFrame, useThree } from "@react-three/fiber"
import { Bloom, EffectComposer, Noise } from "@react-three/postprocessing"
import { useState } from "react"
import { Fog } from "../../effects/fog/Fog"

export const Post: React.FC = () => {
  const [_, setBlah] = useState(0)
  const gl = useThree(state => state.gl)
  const camera = useThree(state => state.camera)
  const light = useThree(state => {
    const light = state.scene.getObjectByName(
      "sun-shadow",
    ) as THREE.DirectionalLight
    return light
  })
  const state = useThree(state => state)
  // workaround for https://github.com/pmndrs/drei/issues/803
  gl.autoClear = true
  const useEffectComposer = true

  useFrame(() => {
    if (light?.shadow?.map?.texture) {
      setBlah(1)
    }
  })

  return useEffectComposer ? (
    <>
      {/* {light?.shadow?.map?.texture && (
        <mesh position={new Vector3(0, 10000, 0)}>
          <planeGeometry args={[10000, 10000]} />
          <meshBasicMaterial map={light.shadow.map.texture} />
        </mesh>
      )} */}
      <EffectComposer>
        {light?.shadow?.map?.texture && (
          <Fog camera={camera} directionalLight={light} />
        )}
        <Bloom luminanceThreshold={0.7} luminanceSmoothing={0.9} height={512} />
        <Noise opacity={0.02} />
        {/* <Vignette eskil={false} offset={0.1} darkness={1.1} /> */}
        {/* <ToneMapping
        blendFunction={BlendFunction.NORMAL} // blend mode
        adaptive={true} // toggle adaptive luminance map usage
        resolution={256} // texture resolution of the luminance map
        middleGrey={0.6} // middle grey factor
        maxLuminance={16.0} // maximum luminance
        averageLuminance={1.0} // average luminance
        adaptationRate={1.0} // luminance adaptation rate
      /> */}
      </EffectComposer>
    </>
  ) : null
}
