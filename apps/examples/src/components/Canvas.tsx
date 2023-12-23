import { Canvas as R3fCanvas, useThree } from "@react-three/fiber"
import React, { Suspense } from "react"
import { Color } from "three"

const Background: React.FC = () => {
  useThree(state => {
    state.scene.background = new Color("#3D4058")
  })
  return null
}

export const Canvas: React.FC<
  React.PropsWithChildren<{ style?: React.CSSProperties }>
> = ({ children, style }) => {
  return (
    <R3fCanvas
      gl={{
        logarithmicDepthBuffer: true,
        antialias: true,
        stencil: true,
        depth: true,
        alpha: true,
      }}
      camera={{
        near: 0.01,
        far: Number.MAX_SAFE_INTEGER,
        position: [-778.8166673411616, 5553.223843712609, 9614.949713806403],
      }}
      shadows="soft"
      shadow-camera-far={1000000}
      shadow-camera-left={-10000}
      shadow-camera-right={10000}
      shadow-camera-top={10000}
      shadow-camera-bottom={-10000}
      style={
        style || {
          position: "absolute",
          top: 0,
          left: 0,
          zIndex: 1,
          background: "#3D4058",
        }
      }
    >
      <Suspense fallback={null}>{children}</Suspense>
      <Background />
    </R3fCanvas>
  )
}
