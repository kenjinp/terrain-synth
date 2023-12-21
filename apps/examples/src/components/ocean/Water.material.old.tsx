import glsl from "glslify"
import React from "react"
import { MeshStandardMaterial, Vector3 } from "three"
import CustomShaderMaterial from "three-custom-shader-material"

import { useFrame, useThree } from "@react-three/fiber"
import CustomShaderMaterialImpl from "three-custom-shader-material/vanilla"
import fragmentShader from "./Water.frag.glsl"
import vertexShader from "./Water.vert.glsl"

export const WaterMaterial: React.FC<{ sunPosition: Vector3 }> = ({
  sunPosition,
}) => {
  const ref = React.useRef<CustomShaderMaterialImpl>(null)
  const camera = useThree(state => state.camera)

  useFrame(({ clock }) => {
    if (!ref.current) return
    ref.current.uniforms.utime.value = clock.getElapsedTime()
    ref.current.uniforms.uCameraPosition.value = camera.position
    ref.current.uniforms.uViewMatrixInverse.value = camera.matrixWorldInverse
    ref.current.uniforms.uProjectionMatrixInverse.value =
      camera.projectionMatrixInverse
    ref.current.uniforms.uSunPosition.value = sunPosition
  })

  return (
    <CustomShaderMaterial
      ref={ref}
      baseMaterial={MeshStandardMaterial}
      fragmentShader={fragmentShader}
      vertexShader={vertexShader}
      uniforms={{
        utime: { value: 0 },
        uCameraPosition: { value: camera.position },
        uViewMatrixInverse: { value: camera.matrixWorldInverse },
        uProjectionMatrixInverse: { value: camera.projectionMatrixInverse },
        uSunPosition: { value: sunPosition },
        depthTexture: { value: null },
        tDepth: { value: null },
      }}
      transparent
      patchMap={{
        csm_NormalColor: {
          "#include <normal_fragment_maps>": glsl`
            #ifdef USE_NORMALMAP_OBJECTSPACE
              normal = csm_NormalColor.xyz * 2.0 -1.0; // overrides both flatShading and attribute normals
              #ifdef FLIP_SIDED
                normal = - normal;
              #endif
              #ifdef DOUBLE_SIDED
                normal = normal * faceDirection;
              #endif
              normal = normalize( normalMatrix * normal );
            #elif defined( USE_NORMALMAP_TANGENTSPACE )
              vec3 mapN = csm_NormalColor.xyz * 2.0 - 1.0;
              mapN.xy *= normalScale;
              normal = normalize( tbn * mapN );
            #elif defined( USE_BUMPMAP )
              normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
            #endif
            `,
        },
      }}
    />
  )
}
