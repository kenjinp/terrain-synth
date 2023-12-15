import { CameraControls } from "@react-three/drei"
import { useFrame, useThree } from "@react-three/fiber"
import * as React from "react"
import { DirectionalLight, Object3D, Vector3 } from "three"

// @ts-ignore
export const windowBounds = "window-bounds"

const lightOrigin = new Vector3(
  6244.923261707597,
  6953.7247328594185,
  6263.770656081942,
)
const lightTarget = new Vector3(0, 0, 0)
const lookAtVector = new Vector3(0, 0, -1)
const lightDirection = new Vector3()
  .subVectors(lightTarget, lightOrigin)
  .normalize()

export const ExampleWrapper: React.FC<
  React.PropsWithChildren<{
    controls?: React.ReactNode
  }>
> = ({
  children,
  controls = (
    <CameraControls
      makeDefault
      maxZoom={1_000}
      minZoom={100}
      maxDistance={15_000}
      minPolarAngle={20 * (Math.PI / 180)}
      maxPolarAngle={80 * (Math.PI / 180)}
    />
  ),
}) => {
  const dirLightRef = React.useRef<DirectionalLight>(null)
  const camera = useThree(state => state.camera)
  const scene = useThree(state => state.scene)
  console.log({ scene })
  // const csm = React.useMemo(() => {
  //   const csm = new CSM({
  //     maxFar: camera.far,
  //     shadowFar: 100_000,
  //     cascades: 4,
  //     shadowMapSize: 1024,
  //     lightDirection,
  //     camera: camera,
  //     parent: scene,
  //   }) as CSMOriginal
  //   const helper = new CSM.Helper(csm)
  //   helper.visible = true
  //   scene.add(helper)
  //   return csm
  // }, [camera, scene])
  // scene.userData["csm"] = csm
  const [target] = React.useState(() => {
    const obj = new Object3D()
    obj.position.copy(lightTarget)
    return obj
  })

  useFrame(() => {
    // if (!dirLightRef.current) return
    // lookAtVector.applyQuaternion(camera.quaternion)
    // lightTarget.copy(camera.position).multiply(lookAtVector)
    // dirLightRef.current.position.copy(camera.position)
    // dirLightRef.current.target.position.copy(lightTarget)
    // csm.update()
  })

  const lightIntensity = 4

  return (
    <>
      {/* <mesh position={[0, 2000, 0]} castShadow receiveShadow>
        <sphereGeometry args={[1000, 32, 32]} />
        <meshStandardMaterial color="red" />
      </mesh> */}
      {/* technique from HamzaKubba */}
      <directionalLight
        ref={dirLightRef}
        name="sun-shadow"
        intensity={lightIntensity * 0.8}
        position={lightOrigin}
        target={target}
        castShadow
        color={"#FFFFFF"}
        shadow-camera-far={15000}
        shadow-camera-left={-8_000}
        shadow-camera-right={8_000}
        shadow-camera-top={8_000}
        shadow-camera-bottom={-8_000}
        shadow-mapSize={[2048, 1024]}
        shadow-bias={-0.0001}
      />
      <directionalLight
        ref={dirLightRef}
        name="sun-no-shadow"
        intensity={lightIntensity * 0.2}
        position={lightOrigin}
        target={target}
        color={"#A9AB74"}
        shadow-camera-far={15000}
        shadow-camera-left={-8_000}
        shadow-camera-right={8_000}
        shadow-camera-top={8_000}
        shadow-camera-bottom={-8_000}
        shadow-mapSize={[2048, 1024]}
        shadow-bias={-0.0001}
      />
      {/* {dirLightRef.current && (
        <directionalLightHelper args={[dirLightRef.current, 10]} />
      )}
      {dirLightRef.current && (
        <cameraHelper args={[dirLightRef.current.shadow.camera]} />
      )} */}
      {/* <ambientLight intensity={0.2} /> */}
      {children}
      {controls}
    </>
  )
}
