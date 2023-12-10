import { FlatWorld as HelloFlatWorld } from "@hello-worlds/planets"
import { FlatWorld } from "@hello-worlds/react"
import { useThree } from "@react-three/fiber"
import { Perf } from "r3f-perf"
import { Color, Euler, Vector3 } from "three"

import { Box, ContactShadows, Grid, Html, useTexture } from "@react-three/drei"
import { useControls } from "leva"
import { useEffect, useMemo, useRef, useState } from "react"
import { match } from "ts-pattern"
import { CloudScroller } from "../../components/clouds/CloudScroller"
import { Compass } from "../../components/compass/Compass"
import { Ocean } from "../../components/ocean/Ocean"
import { Post } from "../../components/post/Post"
import { generateBlendedMaterial } from "../../materials/Ground"
import { MODEL_STATE } from "../../model/Model.utils"
import { useModel } from "../../model/use-model"
import { UI } from "../../tunnel"
import Worker from "./Basic.worker?worker"

async function loadGeneratedImage(
  size: number,
  img: HTMLImageElement,
  // url: string,
  upsample = false,
): Promise<ImageData> {
  // const res = await fetch(url)
  // const blob = await res.blob()
  const image = await createImageBitmap(img)
  const canvas = document.createElement("canvas")
  const context = canvas.getContext("2d")!
  if (upsample) {
    canvas.width = size
    canvas.height = size
    // context.save()
    // context.scale(1, -1)
    context.scale(-1, 1)
    // const blurRadius = 10

    // Apply blur filter
    // context.filter = `blur(${blurRadius}px)`
    // context.drawImage(image, 0, 0, image.width, -image.height)
    // context.drawImage(image, 0, 0)
    context.drawImage(
      image,
      // 5,
      // 5,
      // image.width - 10,
      // image.height - 10,
      0,
      0,
      canvas.width * -1,
      canvas.width,
    )
    // context.restore()

    // Reset filter
    // context.imageSmoothingQuality = "high"
    context.imageSmoothingEnabled = true
    // context.filter = "none"
    // document.body.appendChild(canvas)

    return context.getImageData(0, 0, canvas.width, canvas.width)
  } else {
    canvas.width = image.width
    canvas.height = image.height
    // context.save()
    // context.scale(1, -1)
    // context.scale(-1, 1)
    // const blurRadius = 0

    // Apply blur filter
    // context.filter = `blur(${blurRadius}px)`
    // context.drawImage(image, 0, 0, image.width, -image.height)
    context.drawImage(
      image,
      // 2,
      // 2,
      // image.width - 2,
      // image.height - 2,
      0,
      0,
      canvas.width,
      canvas.width,
    )
    // context.drawImage(image, 0, 0, size * -1, size)
    // context.restore()

    // Reset filter
    context.filter = "none"
    // document.body.appendChild(canvas)

    return context.getImageData(1, 1, image.width, image.height)
  }
}

const worker = () => new Worker()
export default () => {
  const { scaleMax, showPerf } = useControls({
    scaleMax: {
      value: 1000,
      min: 0,
      max: 2000,
    },
    // showHeightmap: true,
    model: false,
    showPerf: false,
  })
  const camera = useThree(state => state.camera)
  const flatWorld = useRef<HelloFlatWorld<any>>(null)
  const { state, run, imageData } = useModel()

  const [uv, sand, grass, rocks] = useTexture([
    "uv.png",
    "beach.png",
    "grass.png",
    "rocks.png",
  ])
  const [terrainData, setTerrainData] = useState<ImageData | null>(null)
  const size = 10_000

  useEffect(() => {
    setTerrainData(imageData)
  }, [imageData])

  useEffect(() => {
    const keyboardListener = (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        handleRegenerateTerrain()
      }
    }
    document.addEventListener("keydown", keyboardListener)
    return () => document.removeEventListener("keydown", keyboardListener)
  })

  const handleRegenerateTerrain = () => {
    run()
  }

  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if (event.key === "c") {
        console.log(camera.position)
      }
    }
    document.addEventListener("keydown", listener)
    return () => document.removeEventListener("keydown", listener)
  }, [])

  const data = useMemo(() => {
    return {
      terrainData,
      scaleMax,
      seed: "Basic Example",
    }
  }, [terrainData, scaleMax])

  const mat = useMemo(() => {
    sand.repeat.set(1000, 1000)
    rocks.repeat.set(1000, 1000)
    grass.repeat.set(1000, 1000)
    return generateBlendedMaterial([
      {
        texture: sand,
      },
      { texture: grass, levels: [10, 12, 20, 30] },
      {
        texture: rocks,
        glsl: "slope > 0.7853981633974483 ? 0.2 : 1.0 - smoothstep(0.47123889803846897, 0.7853981633974483, slope) + 0.2",
      },
    ])
  }, [sand, grass, rocks])

  const depth = scaleMax
  camera.position.set(-778.8166673411616, 5553.223843712609, 9614.949713806403)
  const regenerateDisabled = state !== MODEL_STATE.IDLE
  const label = match(state)
    .with(MODEL_STATE.IDLE, () => "Regenerate Terrain (Enter)")
    .with(MODEL_STATE.RUNNING, () => "Generating Terrain")
    .with(MODEL_STATE.LOADING, () => "Loading Model")
    .run()

  console.log(state, label)

  return (
    <>
      {showPerf && <Perf />}
      <Post />
      <Compass position={new Vector3(size, -depth, -size)} />
      <UI.In>
        <section className="actions-bar">
          <button
            onClick={handleRegenerateTerrain}
            disabled={regenerateDisabled}
          >
            {label}
          </button>
        </section>
      </UI.In>
      <CloudScroller
        bounds={new Vector3(size, depth * 3, size)}
        position={new Vector3(0, scaleMax + depth * 2, 0)}
      />
      <Grid
        position={[0, -depth, 0]}
        cellSize={1_000}
        cellThickness={0.5}
        cellColor={new Color(0x55597b)}
        sectionSize={5_000}
        sectionThickness={1.2}
        sectionColor={new Color(0x55597b)}
        followCamera={false}
        infiniteGrid
        fadeDistance={1_000_000}
        fadeStrength={40}
      />
      <Html
        position={[0, -depth + 100, size / 2 + 500]}
        transform
        scale={[1000, 1000, 1000]}
        rotation={[-90 * (Math.PI / 180), 0, 0]}
        occlude="blending"
      >
        <div>
          <h3 style={{ textShadow: "black 1px 0px 2px" }}>
            HelloWorlds by{" "}
            <a href="https://twitter.com/KennyPirman">kenny.wtf</a>
          </h3>
        </div>
      </Html>
      <Ocean
        position={[0, -depth / 2 + 10, 0]}
        size={[size - 0.1, depth, size - 0.1]}
      />
      <group>
        <Box scale={[size + 100, 1, size + 100]} position-y={0}>
          <meshBasicMaterial color="pink" opacity={0} transparent />
        </Box>
        <ContactShadows
          position={[0, -depth, 0]}
          scale={size * 2}
          far={size * 10}
          blur={3}
          rotation={[Math.PI / 2, 0, 0]}
          color={"black"}
        />
      </group>
      <group
        // Rotate World so it's along the x axis
        rotation={new Euler().setFromVector3(new Vector3(-Math.PI / 2, 0, 0))}
        receiveShadow
      >
        {terrainData && (
          <FlatWorld
            ref={flatWorld}
            size={size}
            minCellSize={64}
            minCellResolution={64}
            lodOrigin={camera.position}
            worker={worker}
            data={data}
            skirtDepth={depth}
            // lodDistanceComparisonValue={1.5}
          >
            <meshPhysicalMaterial
              // baseMaterial={MeshPhysicalMaterial}
              vertexColors

              // map={uv}
              // map={showHeightmap ? newTexture : null}
            />
            {/* <primitive object={mat} /> */}
          </FlatWorld>
        )}
      </group>
    </>
  )
}
