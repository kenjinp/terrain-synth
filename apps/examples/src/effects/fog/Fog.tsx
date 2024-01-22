import { wrapEffect } from "@react-three/postprocessing"
import { Effect, EffectAttribute, WebGLExtension } from "postprocessing"
import {
  Camera,
  DirectionalLight,
  Matrix4,
  Uniform,
  Vector3,
  WebGLRenderTarget,
  WebGLRenderer,
} from "three"
import fragment from "./Fog.glsl"

export interface FogEffectProps {
  camera: Camera
  directionalLight: DirectionalLight
}

// tempValues
const _cameraDirection = new Vector3()
const _position = new Vector3()
const _matrixWorld = new Matrix4()
const _projectionMatrixInverse = new Matrix4()

class FogEffect extends Effect {
  camera: Camera
  directionalLight: DirectionalLight
  constructor({ camera, directionalLight }: FogEffectProps) {
    // camera gets added after construction in effect-composer
    if (camera) {
      camera.getWorldPosition(_position)
      camera.getWorldDirection(_cameraDirection)
    }

    super("FogEffect", fragment, {
      uniforms: new Map<string, Uniform>([
        ["uCameraPosition", new Uniform(_position)],
        ["uCameraWorldDirection", new Uniform(_cameraDirection)],
        ["uViewMatrixInverse", new Uniform(_matrixWorld)],
        ["uProjectionMatrixInverse", new Uniform(_projectionMatrixInverse)],
        ["uDirectionalLightShadow", new Uniform(directionalLight.shadow)],
        [
          "uDirectionalShadowMap",
          new Uniform(directionalLight.shadow.map.texture),
        ],
        [
          "uDirectionalShadowMatrix",
          new Uniform(directionalLight.shadow.matrix),
        ],
        ["uSunPosition", new Uniform(directionalLight.position)],
        ["uTime", new Uniform(0)],
      ]),
      attributes: EffectAttribute.DEPTH,
      extensions: new Set([WebGLExtension.DERIVATIVES]),
    })

    this.camera = camera
    this.directionalLight = directionalLight
  }

  update(
    _renderer: WebGLRenderer,
    _renderTarget: WebGLRenderTarget,
    deltaTime: number,
  ) {
    this.camera.getWorldPosition(_position)
    this.camera.getWorldDirection(_cameraDirection)
    this.uniforms.get("uCameraWorldDirection")!.value = _cameraDirection
    this.uniforms.get("uCameraPosition")!.value = _position
    this.uniforms.get("uTime")!.value += deltaTime
    this.uniforms.get("uViewMatrixInverse")!.value = this.camera?.matrixWorld
    this.uniforms.get("uSunPosition")!.value = this.directionalLight.position
    this.uniforms.get("uDirectionalLightShadow")!.value =
      this.directionalLight.shadow
    this.uniforms.get("uProjectionMatrixInverse")!.value =
      this.camera?.projectionMatrixInverse
  }
}

export const Fog = wrapEffect(FogEffect)
