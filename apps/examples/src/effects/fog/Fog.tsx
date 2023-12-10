import { wrapEffect } from "@react-three/postprocessing"
import { Effect, EffectAttribute, WebGLExtension } from "postprocessing"
import { Camera, Matrix4, Uniform, Vector3 } from "three"
import fragment from "./Fog.glsl"

export interface FogEffectProps {
  camera: Camera
}

// tempValues
const _cameraDirection = new Vector3()
const _position = new Vector3()
const _matrixWorld = new Matrix4()
const _projectionMatrixInverse = new Matrix4()

class FogEffect extends Effect {
  camera: Camera
  constructor({ camera }: FogEffectProps) {
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
      ]),
      attributes: EffectAttribute.DEPTH,
      extensions: new Set([WebGLExtension.DERIVATIVES]),
    })
    this.camera = camera
  }

  update() {
    this.camera.getWorldPosition(_position)
    this.camera.getWorldDirection(_cameraDirection)
    this.uniforms.get("uCameraWorldDirection")!.value = _cameraDirection
    this.uniforms.get("uCameraPosition")!.value = _position
    this.uniforms.get("uViewMatrixInverse")!.value = this.camera?.matrixWorld
    this.uniforms.get("uProjectionMatrixInverse")!.value =
      this.camera?.projectionMatrixInverse
  }
}

export const Fog = wrapEffect(FogEffect)
