import { CameraControls } from "@react-three/drei"
import { useFrame } from "@react-three/fiber"
import React, { useRef } from "react"
import { Vector3 } from "three"
import { radToDeg } from "three/src/math/MathUtils"
import { UI } from "../../tunnel"
import CompassSVG from "./compass.svg?react"

export const Compass: React.FC<{ position: Vector3 }> = ({ position }) => {
  const ref = useRef<HTMLDivElement>(null)

  useFrame(state => {
    if (!ref.current) return
    const controls = state.controls as unknown as CameraControls
    if (!controls) return
    ref.current.style.transformOrigin = "center"
    ref.current.style.transform = `rotate3d(1, 0, 0, ${radToDeg(
      controls.polarAngle || 0,
    )}deg) rotate3d(0, 0, 1, ${radToDeg(controls.azimuthAngle || 0)}deg)`
  })

  return (
    <>
      <UI.In>
        <div style={{ padding: "2em" }}>
          <div
            ref={ref}
            style={{
              opacity: 0.8,
              width: "fit-content",
            }}
          >
            <CompassSVG />
          </div>
        </div>
      </UI.In>
      {/* <Html
        transform
        position={position}
        scale={[1000, 1000, 1000]}
        rotation={[90 * (Math.PI / 180), 0, 180 * (Math.PI / 180)]}
      >
        <div style={{ opacity: 0.5 }}>
          <CompassSVG />
        </div>
      </Html> */}
    </>
  )
}
