import { Chunk } from "@hello-worlds/planets"
import { useFlatWorldChunks } from "@hello-worlds/react"
import { FC, useEffect, useMemo, useState } from "react"
import { Texture } from "three"
import { ScatterMap } from "./maps"

export interface ChunkScatterProps {
  targetLOD: number
}

const makeChunkHash = (chunk: Chunk) => {
  return `${chunk.width}-${chunk.offset.toArray().join(",")}`
}

const createImageFromUint8Array = (data: Uint8Array, width: number) => {
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = width
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  const imageData = ctx.createImageData(width, width)
  imageData.data.set(data)
  ctx.putImageData(imageData, 0, 0)
  canvas.style.position = "absolute"
  canvas.style.zIndex = "1000"
  canvas.style.backgroundColor = "red"
  document.body.appendChild(canvas)
  // Create an HTML image element
  const imageElement = new Image()

  // Set the image source to the canvas data URL
  imageElement.src = canvas.toDataURL()
  // imageElement.id = "ocean heights"
  // document.body.appendChild(imageElement)

  return imageElement
}

export const ChunkScatter: FC<ChunkScatterProps> = ({ targetLOD }) => {
  const [_, rerender] = useState(0)
  const chunks = useFlatWorldChunks()
  const [targetChunks, setTargetChunks] = useState<Chunk[]>([])
  const [scatterMapMap] = useState<Map<string, ScatterMap>>(new Map())

  useEffect(() => {
    const newTargetChunks: Chunk[] = []

    const filteredChunks = chunks.filter(chunk => chunk.lodLevel === targetLOD)

    filteredChunks.forEach(chunk => {
      const hash = makeChunkHash(chunk)
      const targetChunk = targetChunks.find(
        chunk => makeChunkHash(chunk) === hash,
      )
      if (!targetChunk) {
        newTargetChunks.push(chunk)
      }
    })
    const newArray = [...targetChunks, ...newTargetChunks]

    if (newArray.length !== targetChunks.length) {
      setTargetChunks(newArray)
    }
  }, [chunks])

  // so now we have the heightmaps, we can iterate over them and create a scatter map for each one
  useEffect(() => {
    console.log("hello")
    targetChunks.map(chunk => {
      if (!chunk.heightmap) return
      const chunkKey = makeChunkHash(chunk)
      if (scatterMapMap.has(chunkKey)) return
      const scatterMap = new ScatterMap(
        makeChunkHash(chunk),
        chunk.heightmap,
        chunk.minHeight,
        chunk.maxHeight,
        {
          heightMask: {
            min: 0,
            max: 200,
          },
        },
      )
      scatterMap.points = scatterMap.generatePoints(chunk, 2_000)
      scatterMapMap.set(chunkKey, scatterMap)
      // createImageFromUint8Array(
      //   scatterMap.scatterMap!,
      //   Math.sqrt(chunk.heightmap.byteLength / 4),
      // )
      // console.log({ points: scatterMap.generatePoints(chunk) })
      rerender(prev => prev + 1)
    })
  }, [targetChunks])

  const vals = useMemo(() => Array.from(scatterMapMap.entries()), [_])

  return (
    <>
      {vals.map(([key, scatterMap], index) => {
        const width = Math.sqrt(scatterMap.heightmap.byteLength / 4)
        console.log({ width })
        const texture = new Texture(
          createImageFromUint8Array(scatterMap.scatterMap, width),
        )
        return (
          <group key={key}>
            <mesh>
              <boxGeometry args={[width, 1, width]} />
              <meshStandardMaterial map={texture} />
            </mesh>

            {/* <Scatter positions={scatterMap.points!} /> */}
          </group>
        )
      })}
    </>
  )
}
