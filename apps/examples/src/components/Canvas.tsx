import { Canvas as R3fCanvas } from "@react-three/fiber"
import { Suspense } from "react"
import { PCFSoftShadowMap } from "three"
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
      }}
      camera={{
        near: 0.01,
        far: Number.MAX_SAFE_INTEGER,
      }}
      shadows={{ type: PCFSoftShadowMap }}
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
    </R3fCanvas>
  )
}
