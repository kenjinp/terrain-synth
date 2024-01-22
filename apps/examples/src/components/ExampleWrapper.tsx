import { CameraControls } from "@react-three/drei"
import * as React from "react"
import { DirectionalLight, Object3D, Vector3 } from "three"

// @ts-ignore
export const windowBounds = "window-bounds"

const lightOrigin = new Vector3(
  // 6244.923261707597,
  // 6953.7247328594185,
  // 6263.770656081942,
  // 9504.849326170277,
  // 1460.8306729382275,
  // 5262.743402506646,
  7728.335020729982,
  4494.66353967614,
  6630.455936901849,

  // 9115.213935946103,
  // 1516.239736791226,
  // -11869.589999777365,
)
const lightTarget = new Vector3(0, 0, 0)

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

  const [target] = React.useState(() => {
    const obj = new Object3D()
    obj.position.copy(lightTarget)
    return obj
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
        intensity={lightIntensity * 0.7}
        position={lightOrigin}
        target={target}
        castShadow
        color={"#FFFFFF"}
        shadow-camera-far={18000}
        shadow-camera-left={-8_000}
        shadow-camera-right={8_000}
        shadow-camera-top={8_000}
        shadow-camera-bottom={-8_000}
        shadow-mapSize={[2048, 1024]}
        shadow-bias={-0.003}
      />
      <directionalLight
        ref={dirLightRef}
        name="sun-no-shadow"
        intensity={lightIntensity * 0.3}
        position={lightOrigin}
        target={target}
        color={"#A9AB74"}
      />
      {children}
      {controls}
    </>
  )
}
