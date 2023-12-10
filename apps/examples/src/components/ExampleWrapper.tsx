import { CameraControls } from "@react-three/drei"
import * as React from "react"

export const windowBounds = "window-bounds"

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
  return (
    <>
      <directionalLight intensity={2} />
      <ambientLight intensity={1} />
      {children}
      {controls}
    </>
  )
}
