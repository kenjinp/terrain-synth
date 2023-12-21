import { useFBO, useTexture } from "@react-three/drei"
import { useFrame, useThree } from "@react-three/fiber"
import { useControls } from "leva"
import React from "react"
import {
  Mesh,
  MeshStandardMaterial,
  RepeatWrapping,
  Vector2,
  Vector3,
} from "three"

import CustomShaderMaterial from "three-custom-shader-material"
import CustomShaderMaterialImpl from "three-custom-shader-material/vanilla"
import fragmentShader from "./Water.frag.glsl"
import vertexShader from "./Water.vert.glsl"

export const WaterMaterial: React.FC<{ sunPosition: Vector3; mesh: Mesh }> = ({
  sunPosition,
  mesh,
}) => {
  const ref = React.useRef<CustomShaderMaterialImpl>(null)

  const { absorbStrength, shine, diffuse } = useControls({
    absorbStrength: { value: 2.0, min: 0.0, max: 100.0, step: 0.1 },
    shine: { value: 150, min: 0.0, max: 200.0, step: 0.1 },
    diffuse: { value: 0.1, min: 0.0, max: 10.0, step: 0.1 },
  })

  const dudvTexture = useTexture("/water_dudv.png")
  const normalTexture = useTexture("/water_normal.jpg")
  normalTexture.wrapS = normalTexture.wrapT = RepeatWrapping

  const { viewport, size, camera } = useThree()

  const FBOSettings = {
    depth: true,
  }
  const renderTarget = useFBO(FBOSettings)

  useFrame(state => {
    if (!ref.current) return
    const { gl, scene, camera, clock } = state
    mesh.visible = false
    gl.setRenderTarget(renderTarget)
    gl.render(scene, camera)
    ref.current.uniforms.u_sceneTexture.value = renderTarget.texture
    ref.current.uniforms.u_depthTexture.value = renderTarget.depthTexture
    gl.setRenderTarget(null)
    mesh.visible = true

    ref.current.uniforms.u_lightPos.value = sunPosition
    ref.current.uniforms.u_time.value = clock.elapsedTime
    ref.current.uniforms.u_winRes.value =
      ref.current.uniforms.u_winRes.value.set(
        size.width * viewport.dpr,
        size.height * viewport.dpr,
      )
    ref.current.uniforms.u_near.value = camera.near
    ref.current.uniforms.u_far.value = camera.far
    ref.current.uniforms.u_absorbStrength.value = absorbStrength
    ref.current.uniforms.u_shine.value = shine
    ref.current.uniforms.u_diffuse.value = diffuse
  })
  return (
    <>
      {/* Debug Plane to visualize render target! */}
      {/* <mesh position={new Vector3(0, 10000, 0)}>
        <planeGeometry args={[10000, 10000]} />
        <meshBasicMaterial map={renderTarget.depthTexture} />
      </mesh> */}
      <CustomShaderMaterial
        ref={ref}
        baseMaterial={MeshStandardMaterial}
        fragmentShader={fragmentShader}
        vertexShader={vertexShader}
        transparent
        uniforms={{
          // ...UniformsLib.lights,
          u_lightPos: { value: sunPosition },
          u_sceneTexture: { value: null },
          u_depthTexture: { value: null },
          u_dudvTexture: { value: dudvTexture },
          u_normalTexture: { value: normalTexture },
          u_winRes: {
            value: new Vector2(
              size.width * viewport.dpr,
              size.height * viewport.dpr,
            ),
          },
          u_near: { value: camera.near },
          u_far: { value: camera.far },
          u_time: { value: 0 },
          u_absorbStrength: { value: absorbStrength },
          u_shine: { value: shine },
          u_diffuse: { value: diffuse },
        }}
      />
    </>
  )
}
