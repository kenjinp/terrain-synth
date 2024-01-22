import { FlatWorld as HelloFlatWorld } from "@hello-worlds/planets"
import { FlatWorld } from "@hello-worlds/react"
import { useThree } from "@react-three/fiber"
import { Color, Euler, Vector3 } from "three"

import { Grid } from "@react-three/drei"
import React, { useEffect, useMemo, useRef } from "react"
import { match } from "ts-pattern"
import { CloudScroller } from "../../components/clouds/CloudScroller"
import { Compass } from "../../components/compass/Compass"
import { Ocean } from "../../components/ocean/Ocean"
import { Post } from "../../components/post/Post"
import { useSeed } from "../../hooks/use-seed"
import { MODEL_STATE } from "../../model/Model.gan"
import { MODEL_STRATEGIES, useModel } from "../../model/use-model"
import { UI } from "../../tunnel"
import { BIOMES } from "./Home.biomes"
import Worker from "./Home.worker?worker"

const worker = () => new Worker()

export default function HomePage() {
  const { seed } = useSeed()
  return seed ? <Home seed={seed} /> : null
}

export const Home: React.FC<{
  seed: string
  useNoise?: boolean
  useInterpolation?: boolean
  /**
   * @min 0
   * @max 2000
   */
  scaleMax?: number
  strategy?: keyof typeof MODEL_STRATEGIES
}> = ({
  seed,
  strategy = "WGAN",
  useNoise = true,
  useInterpolation = true,
  scaleMax = 700,
}) => {
  const { setRandomSeed } = useSeed()
  const camera = useThree(state => state.camera)
  const flatWorld = useRef<HelloFlatWorld<unknown>>(null)
  const { state, result } = useModel(strategy, seed)

  const size = 10_000

  const handleRegenerateTerrain = () => {
    setRandomSeed()
  }

  useEffect(() => {
    const keyboardListener = (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        handleRegenerateTerrain()
      }
    }
    document.addEventListener("keydown", keyboardListener)
    return () => document.removeEventListener("keydown", keyboardListener)
  })

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
      terrainData: result?.terrainData,
      oceanData: result?.oceanData,
      scaleMax,
      useNoise,
      useInterpolation,
      seed,
    }
  }, [result, scaleMax, useNoise, useInterpolation, seed])

  const depth = scaleMax
  const regenerateDisabled = state !== MODEL_STATE.IDLE
  const label = match(state)
    .with(MODEL_STATE.IDLE, () => "Regenerate Terrain (Enter)")
    .with(MODEL_STATE.RUNNING, () => "Generating Terrain")
    .with(MODEL_STATE.LOADING, () => "Loading Model")
    .otherwise(() => "Unknown State")

  return (
    <>
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
      <Ocean
        position={[0, -depth / 2 + 5, 0]}
        size={[size - 0.1, depth, size - 0.1, 1000, 10, 1000]}
      />
      <group
        // Rotate World so it's along the x axis
        rotation={new Euler().setFromVector3(new Vector3(-Math.PI / 2, 0, 0))}
        receiveShadow
      >
        {result && (
          <FlatWorld
            ref={flatWorld}
            size={size}
            minCellResolution={128}
            minCellSize={64 * 6}
            lodOrigin={camera.position}
            worker={worker}
            data={data}
            skirtDepth={depth}
          >
            <meshStandardMaterial vertexColors />
          </FlatWorld>
        )}
      </group>
    </>
  )
}
