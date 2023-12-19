import { FlatWorld as HelloFlatWorld } from "@hello-worlds/planets"
import { FlatWorld } from "@hello-worlds/react"
import { useThree } from "@react-three/fiber"
import { Perf } from "r3f-perf"
import { Color, Euler, MeshPhysicalMaterial, Vector3 } from "three"

import { Box, ContactShadows, Grid } from "@react-three/drei"
import { useControls } from "leva"
import { useEffect, useMemo, useRef, useState } from "react"
import { match } from "ts-pattern"
import { CloudScroller } from "../../components/clouds/CloudScroller"
import { Compass } from "../../components/compass/Compass"
import { Ocean } from "../../components/ocean/Ocean"
import { Post } from "../../components/post/Post"
import { MODEL_STATE } from "../../model/Model.gan"
import {
  MODEL_STRATEGIES,
  MODEL_STRATEGY_NAMES,
  useModel,
} from "../../model/use-model"
import { UI } from "../../tunnel"
import { BIOMES } from "./Basic.biomes"
import {
  createImageElementFromImageData,
  processImageData,
} from "./Basic.image"
import Worker from "./Basic.worker?worker"

const worker = () => new Worker()
export default () => {
  const { scaleMax, showPerf, useNoise, useInterpolation, strategy } =
    useControls({
      scaleMax: {
        value: 500,
        min: 0,
        max: 2000,
      },
      // showHeightmap: true,
      model: false,
      showPerf: false,
      useNoise: true,
      useInterpolation: true,
      strategy: {
        options: MODEL_STRATEGY_NAMES,
      },
    })
  const camera = useThree(state => state.camera)
  const scene = useThree(state => state.scene)
  const flatWorld = useRef<HelloFlatWorld<any>>(null)
  const { state, run, imageData } = useModel(
    strategy as keyof typeof MODEL_STRATEGIES,
  )

  // const [uv, sand, grass, rocks] = useTexture([
  //   "uv.png",
  //   "beach.png",
  //   "grass.png",
  //   "rocks.png",
  // ])
  const [terrainData, setTerrainData] = useState<ImageData | null>(null)
  const [oceanData, setOceanData] = useState<ImageData | null>(null)
  const size = 10_000

  useEffect(() => {
    if (!imageData) return
    const oceanData = processImageData(imageData)
    setOceanData(oceanData)
    createImageElementFromImageData(oceanData)
    setTerrainData(imageData)
    createImageElementFromImageData(imageData)
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
        console.log(camera.position.toArray())
      }
    }
    document.addEventListener("keydown", listener)
    return () => document.removeEventListener("keydown", listener)
  }, [])

  const data = useMemo(() => {
    return {
      biome: BIOMES.SIM_CITY,
      terrainData,
      oceanData,
      scaleMax,
      useNoise,
      useInterpolation,
      seed: "Basic Example",
    }
  }, [terrainData, oceanData, scaleMax, useNoise, useInterpolation])

  // const mat = useMemo(() => {
  //   sand.repeat.set(1000, 1000)
  //   rocks.repeat.set(1000, 1000)
  //   grass.repeat.set(1000, 1000)
  //   return generateBlendedMaterial([
  //     {
  //       texture: sand,
  //     },
  //     { texture: grass, levels: [10, 12, 20, 30] },
  //     {
  //       texture: rocks,
  //       glsl: "slope > 0.7853981633974483 ? 0.2 : 1.0 - smoothstep(0.47123889803846897, 0.7853981633974483, slope) + 0.2",
  //     },
  //   ])
  // }, [sand, grass, rocks])

  const mat = useMemo(() => {
    const csm = scene.userData["csm"]
    // const material
    const material = new MeshPhysicalMaterial({ vertexColors: true })
    // if (csm) {
    //   csm.setupMaterial(material)
    //   console.log("set up material with csm")
    // }
    return material
  }, [scene])

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
        sectionThickness={1}
        sectionColor={new Color(0x55597b)}
        followCamera={false}
        infiniteGrid
        fadeDistance={1_000_000}
        fadeStrength={40}
      />

      {/* <Html
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
      </Html> */}
      <Ocean
        position={[0, -depth / 2 + 5, 0]}
        size={[size - 0.1, depth, size - 0.1]}
      />
      <group visible={false}>
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
        // visible={false}
      >
        {/* <ParticleField /> */}

        {terrainData && (
          <FlatWorld
            ref={flatWorld}
            size={size}
            minCellResolution={128}
            minCellSize={64 * 6}
            // minCellResolution={8}
            lodOrigin={camera.position}
            worker={worker}
            data={data}
            skirtDepth={depth}
          >
            {/* <meshPhysicalMaterial
              // baseMaterial={MeshPhysicalMaterial}
              vertexColors

              // map={uv}
              // map={showHeightmap ? newTexture : null}
            /> */}
            {/* <meshNormalMaterial /> */}
            <primitive object={mat} />
          </FlatWorld>
        )}
      </group>
    </>
  )
}
