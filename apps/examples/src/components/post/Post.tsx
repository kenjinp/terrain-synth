import { useThree } from "@react-three/fiber"
import { Bloom, EffectComposer, Noise } from "@react-three/postprocessing"
import { Fog } from "../../effects/fog/Fog"
import { useSun } from "../../hooks/use-sun"

export const Post: React.FC = () => {
  const gl = useThree(state => state.gl)
  const camera = useThree(state => state.camera)
  const light = useSun()
  // workaround for https://github.com/pmndrs/drei/issues/803
  gl.autoClear = true
  const useEffectComposer = true

  return useEffectComposer ? (
    <>
      <EffectComposer>
        {light?.shadow?.map?.texture && (
          <Fog camera={camera} directionalLight={light} />
        )}
        <Bloom luminanceThreshold={0.3} luminanceSmoothing={0.8} height={512} />
        <Noise opacity={0.016} />
      </EffectComposer>
    </>
  ) : null
}
