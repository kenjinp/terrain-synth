import { useThree } from "@react-three/fiber"
import { Bloom, EffectComposer, Noise } from "@react-three/postprocessing"

export const Post: React.FC = () => {
  const gl = useThree(state => state.gl)
  const camera = useThree(state => state.camera)
  // workaround for https://github.com/pmndrs/drei/issues/803
  gl.autoClear = true
  return (
    <EffectComposer>
      {/* <DepthOfField
        focusDistance={0}
        focalLength={0.02}
        bokehScale={2}
        height={300}
      /> */}
      <Bloom luminanceThreshold={0.5} luminanceSmoothing={0.9} height={512} />
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
  )
}
