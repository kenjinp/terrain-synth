import { createRandomSeed } from "@hello-worlds/core"

// Box-Muller transform
export function generateStandardNormalArray(
  result: number[],
  size: number,
  seed?: string,
): number[] {
  const random = createRandomSeed(
    seed || Math.random().toString(36).substring(6),
  )
  for (let i = 0; i < size; i += 2) {
    const u1 = random.next()
    const u2 = random.next()

    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2)
    result[i] = z1
    result[i + 1] = z2
  }

  // If x is odd, remove the last element
  if (size % 2 !== 0) {
    result.pop()
  }

  return result
}

export function generateRandomArray(
  result: number[],
  size: number,
  seed: string,
): number[] {
  const random = createRandomSeed(seed)
  for (let i = 0; i < size; i++) {
    // random value between -1 and 1
    result[i] = random.next() * 2 - 1
  }

  return result
}
