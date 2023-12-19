import {
  MeshStandardMaterialProps,
  extend,
  useFrame,
  useThree,
} from "@react-three/fiber"
import glsl from "glslify"
import { useEffect, useMemo, useRef } from "react"
import * as THREE from "three"
import CustomShaderMaterial from "three-custom-shader-material/vanilla"
import { Reflector, Refractor } from "three/examples/jsm/Addons.js"

export type waterMaterialOptions = MeshStandardMaterialProps & {
  surfaceNormal1: THREE.Texture
  surfaceNormal2?: THREE.Texture
}

function updateTextureMatrix(camera, scope, textureMatrix) {
  textureMatrix.set(
    0.5,
    0.0,
    0.0,
    0.5,
    0.0,
    0.5,
    0.0,
    0.5,
    0.0,
    0.0,
    0.5,
    0.5,
    0.0,
    0.0,
    0.0,
    1.0,
  )

  textureMatrix.multiply(camera.projectionMatrix)
  textureMatrix.multiply(camera.matrixWorldInverse)
  textureMatrix.multiply(scope.matrixWorld)
}

class WaterMaterial extends CustomShaderMaterial {
  constructor(props: any) {
    super({
      baseMaterial: THREE.MeshStandardMaterial,
      uniforms: {
        depthTexture: { value: null },
        surfaceNormal1: { value: props.surfaceNormal1 },
        surfaceNormal2: { value: props.surfaceNormal2 },
        projectionMatrixInverse: {
          value: props.projectionMatrixInverse || null,
        },
        viewMatrixInverse: { value: props.viewMatrixInverse || null },
        reflectionMap: { value: null },
        refractionMap: { value: null },
        textureMatrix: { value: null },
        playerPosition: { value: null },
        time: { value: 0.1 },
      },
      vertexShader: glsl`
        varying vec4 vPos;
        varying vec3 uvz;
        varying vec3 fragWorldPosition;
        uniform mat4 textureMatrix;
        varying vec4 vCoord;
        varying vec3 vToEye;
        varying vec3 worldNormal;
        void main(){
          uvz = vec3(uv, 0.0);
          vCoord = textureMatrix * vec4( position, 1.0 );
          vPos = (projectionMatrix * modelViewMatrix * vec4( position, 1.0 ));
          vec4 worldPoz = modelMatrix * vec4( position, 1.0 );
			    vToEye = worldPoz.xyz - cameraPosition;
          worldNormal = normalize(normalMatrix * normal);
          fragWorldPosition = (modelMatrix * vec4(position.xyz + vec3(0,0,0.0), 1.0)).xyz;
        }
      `,
      fragmentShader: glsl`
        varying vec3 uvz;
        uniform sampler2D depthTexture;
        uniform sampler2D surfaceNormal1;
        uniform sampler2D surfaceNormal2;
        uniform sampler2D reflectionMap;
        uniform sampler2D refractionMap;
        uniform vec3 playerPosition;
        uniform float time;
        varying vec4 vPos;
        varying vec3 fragWorldPosition;
        varying vec4 vCoord;
        varying vec3 vToEye;
        varying vec3 worldNormal;

        // used to control normal
        vec4 csm_NormalColor;

        
        // near and far
        float n = 0.1;
        float f = 2000.0;

        uniform mat4 projectionMatrixInverse;
        uniform mat4 viewMatrixInverse;

        vec4 blend_normals(vec4 n1, vec4 n2){
          vec3 t = n1.xyz*vec3( 2,  2, 2) + vec3(-1, -1,  0);
          vec3 u = n2.xyz*vec3(-2, -2, 2) + vec3( 1,  1, -1);
          vec3 r = t*dot(t, u) /t.z -u;
          return vec4((r), 1.0) * 0.5 + 0.5;
        }

        float readDepth( sampler2D depthSampler, vec2 coord ) {
          float fragCoordZ = texture2D( depthSampler, coord ).x;
          return f * n / ((n-f) * fragCoordZ + f);
        }

        float calculateWaterTransparency(float depth, float turbidity) {
          // Adjust these parameters based on your scene and artistic preferences
          float maxDepth = 10.0;  // Maximum depth before complete opacity
          float turbidityFactor = 0.25;  // Adjust based on the turbidity of the water
      
          // Linear attenuation
          float linearAttenuation = 0.8 - min(depth / maxDepth, 1.0);
      
          // Exponential attenuation
          float exponentialAttenuation = exp(-depth * turbidity * turbidityFactor);
      
          // Combined transparency
          float transparency = linearAttenuation * exponentialAttenuation;
      
          // Ensure transparency is clamped between 0.0 and 1.0
          return clamp(transparency, 0.0, 1.0);
        }

        // magical function, requires projectionMatrixInverse and viewMatrixInverse uniforms
        vec3 worldCoordinatesFromDepth(float depth, vec2 uv) {
          float z = depth * 2.0 - 1.0;
      
          vec4 clipSpaceCoordinate = vec4(uv * 2.0 - 1.0, z, 1.0);
          vec4 viewSpaceCoordinate = projectionMatrixInverse * clipSpaceCoordinate;
      
          viewSpaceCoordinate /= viewSpaceCoordinate.w;
      
          vec4 worldSpaceCoordinates = viewMatrixInverse * viewSpaceCoordinate;
      
          return worldSpaceCoordinates.xyz;
        }
          
        void main() {
          // horizon highlight
          float pixelDistance = abs(gl_FragCoord.z) / gl_FragCoord.w;
          csm_DiffuseColor.rgb *= mix(1.0, 1.1, pixelDistance / 20.);

          // two scrolling normal maps blended together gives the impression of waves
          vec4 a = texture2D(surfaceNormal2, uvz.xy * 1.0 * vec2(50.0,100.0) + vec2(time,0) );
          vec4 b = texture2D(surfaceNormal2, uvz.xy * 4.0 * vec2(75.0,100.0) - vec2(0,-time*2.0));
          // 1.0
          csm_NormalColor = a;
          csm_NormalColor = blend_normals(
            a * 1.25,
            b * 1.5
          );
          
          // bump the normal highlights a bit
          float nfresnel = 1.0 - dot( normalize( csm_NormalColor.xyz ), normalize( vec3( 0.0, -0.1, 1.0 ) ) );
          csm_DiffuseColor.rgb += (nfresnel * nfresnel ) / 15.0;

          // calculate the screen space uv
          vec2 vCoords = vPos.xy / vPos.w / 2.0 + 0.5;
          vec2 suv = fract( vCoords );

          // world coordinates
          float depth = texture( depthTexture, suv ).x;
          vec3 worldPosition = worldCoordinatesFromDepth(depth, suv);
          

          // micro waves
          vec3 center = playerPosition - (playerPosition- cameraPosition)/5.0;
          float dist = clamp(0.6 - length(fragWorldPosition.xz - center.xz)/4.0, 0.0, 1.0);
          float playerWaveFilter = clamp(1.0-dist*4.0, 0.2, 0.5);
          worldPosition.y -= sin(time*80.0)/ 40.0 * playerWaveFilter;
          worldPosition.y -= (cos(worldPosition.x +time*80.0) + sin(worldPosition.y +time*80.0)) / 40.0 * playerWaveFilter;

          // THIS IS A MAGIC NUMBER, it's the height of the water in my scene - should be a uniform
          float height = 9.0; 
          if(worldPosition.y > height-0.3){
            // cut off the top of the water to avoid flat edges if sin value is positive
            csm_DiffuseColor.a = 0.0;
          }
          if(worldPosition.y > height-0.31){
            // simple solid white foam
            csm_DiffuseColor.rgb = vec3(1.0, 1.0, 1.0);
          }

          // reflection and refraction
          float edgeFallOff = clamp((8.75-worldPosition.y)/2.0, 0.0,0.5);
          vec3 coord = vCoord.xyz / vCoord.w;
          float distanceFalloff = clamp(5.0 / length(cameraPosition - fragWorldPosition), 0.0, 1.0);
          float distortion = 0.1 * clamp((distanceFalloff * edgeFallOff) - dist, 0.0, 1.0); 
          vec2 ruv = coord.xy + csm_NormalColor.xz * vec2( 0.01, distortion);
          vec4 reflectColor = texture2D(reflectionMap, vec2( 1.0 - ruv.x, ruv.y ));
          float iorRatio = 1.0/1.31 ;
          ruv = coord.xy;
          vec3 refractVector = refract( normalize(fragWorldPosition.xyz - cameraPosition.xyz), (csm_NormalColor.xyz) * 1.0, iorRatio );
          vec4 refractColor = texture2D( refractionMap, ruv - refractVector.xy/200.0);

          // increase water transparency based on water depth
          float d = texture( depthTexture, ruv - refractVector.xy/200.0 ).x;
          vec3 w = worldCoordinatesFromDepth(d, ruv - refractVector.xy/200.0);
          float dp2 = length(abs(w- fragWorldPosition));
          csm_DiffuseColor.rgb = mix(refractColor.rgb, csm_DiffuseColor.rgb, clamp(1.0-calculateWaterTransparency(dp2, 0.5), 0.0,1.0));

          // make water more or less transparent based on angle of view
          float fresnel = clamp(
            0.5 -
            dot( normalize(csm_NormalColor.xyz), normalize( vec3( 0.0, 0.0, 1.0 ) ) ) * 0.15 -
            dot( normalize(worldPosition.xyz - cameraPosition.xyz), normalize( vec3( 0.0, -1.0, 0.0 ) ) ) * 0.65
            , 0.0, 1.0);
          csm_DiffuseColor.rgb = mix(csm_DiffuseColor.rgb, reflectColor.rgb, fresnel);
          
        }
      `,
      patchMap: {
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
      },
    })

    setInterval(() => {
      this.uniforms.time.value += 0.0002
    }, 10)
  }
}

export function Water(props: any) {
  extend({ WaterMaterial })

  const ref = useRef<WaterMaterial>(null)
  const reflector = useRef<Reflector>(null)
  const refractor = useRef<Refractor>(null)
  const textureMatrix = new THREE.Matrix4()
  const clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), 8.9)

  // defines are passed to shaders
  const defines = useMemo(() => {
    const temp = {} as { [key: string]: string }
    return temp
  }, [])

  //prevents constructor from running on prop changes, key is used to trigger reconstruction
  const args = useMemo(() => {
    return [{ ...props }]
  }, [])

  const [target] = useMemo(() => {
    const target = new THREE.WebGLRenderTarget(
      window.innerWidth * window.devicePixelRatio,
      window.innerHeight * window.devicePixelRatio,
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        stencilBuffer: false,
        depthBuffer: true,
        depthTexture: new THREE.DepthTexture(
          window.innerWidth,
          window.innerHeight,
        ),
      },
    )
    return [target]
  }, [])

  useFrame(state => {
    if (ref.current && props.mesh) {
      props.mesh.visible = false
      props.mesh.material.fog = false
      // reflector.current.onBeforeRender( state.gl, state.scene, state.camera );
      state.gl.setRenderTarget(target)
      state.gl.clippingPlanes = [clip]
      state.gl.render(state.scene, state.camera)
      state.gl.clippingPlanes = []
      props.mesh.visible = true
      if (ref.current) {
        ref.current.uniforms["depthTexture"].value = target.depthTexture
        ref.current.uniforms["refractionMap"].value = target.texture
        ref.current.uniforms["projectionMatrixInverse"].value =
          state.camera.projectionMatrixInverse
        ref.current.uniforms["viewMatrixInverse"].value =
          state.camera.matrixWorld
        // todo, don't use window
        ref.current.uniforms["playerPosition"].value = new THREE.Vector3() //window.playerPosition
      }
      state.gl.setRenderTarget(null)
    }
  })

  const { scene } = useThree()

  useEffect(() => {
    // todo: should this be a prop or no?
    if (props.mesh) {
      //@ts-ignore
      reflector.current = new Reflector(props.mesh.geometry, {
        clipBias: 0.0,
        textureWidth: window.innerWidth / 4.0,
        textureHeight: window.innerHeight / 4.0,
        color: 0x0b5189,
      })
      reflector.current.matrixAutoUpdate = false
      if (ref.current) {
        ref.current.uniforms["reflectionMap"].value =
          reflector.current.getRenderTarget().texture
        ref.current.uniforms["textureMatrix"].value = textureMatrix
      }

      props.mesh.onBeforeRender = (renderer, scene, camera) => {
        updateTextureMatrix(camera, props.mesh, textureMatrix)
        const scope = props.mesh
        scope.visible = false
        reflector.current.matrixWorld.copy(scope.matrixWorld)
        reflector.current.onBeforeRender(renderer, scene, camera)
        scope.visible = true
      }

      return () => {
        reflector.current && scene.remove(reflector.current)
      }
    }
  }, [props.mesh])

  //@ts-ignore
  return <waterMaterial ref={ref} {...props} args={args} defines={defines} />
}
