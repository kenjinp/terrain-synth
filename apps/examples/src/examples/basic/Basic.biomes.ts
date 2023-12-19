import { Lerp, LinearSpline } from "@hello-worlds/planets"
import { Color } from "three"

export enum BIOMES {
  SIM_CITY,
  TROPICAL,
  OCEAN,
}

export const biomeNames = {
  [BIOMES.SIM_CITY]: "Sim City",
  [BIOMES.TROPICAL]: "Tropical",
}

interface ColorElevation {
  color: string
  elevation: number
}

const oceanColors: ColorElevation[] = [
  { color: "#80FFE0", elevation: 0.0 },
  { color: "#95EBCE", elevation: 0.1 },
  { color: "#4C7788", elevation: 0.5 },
]

const simCityColors: ColorElevation[] = [
  { color: "#d9c9bb", elevation: 0.0 },
  { color: "#e1bd9c", elevation: 0.02 },
  { color: "#494f2b", elevation: 0.025 },
  { color: "#6f6844", elevation: 0.1 },
  { color: "#927e59", elevation: 0.3 },
  { color: "#816653", elevation: 0.6 },
  { color: "#70666d", elevation: 0.65 },
  { color: "#ffffff", elevation: 0.7 },
]

const tropicalColors: ColorElevation[] = [
  { color: "#d9c9bb", elevation: 0.0 }, // Beach
  { color: "#e1bd9c", elevation: 0.025 }, // High Beach
  { color: "#657245", elevation: 0.03 }, // Low Grass
  { color: "#494f2b", elevation: 0.2 }, // Foresty Bits
  { color: "#323c39", elevation: 0.55 }, // Steep Hills
  { color: "#816653", elevation: 0.6 }, // Rocky stuff
  { color: "#927e59", elevation: 0.65 }, // Steep Hills
  { color: "#70666d", elevation: 0.73 }, // Glacier
  { color: "#ffffff", elevation: 0.74 }, // Snow
]

const colorLerp: Lerp<THREE.Color> = (
  t: number,
  p0: THREE.Color,
  p1: THREE.Color,
) => {
  const c = p0.clone()
  return c.lerp(p1, t)
}

function createColorSplineFromColorElevation(colorElevation: ColorElevation[]) {
  const colorSpline = new LinearSpline<Color>(colorLerp)
  colorElevation.forEach(({ color, elevation }) => {
    colorSpline.addPoint(elevation, new Color(color))
  })
  return colorSpline
}

export const biomeColorSplineMap = {
  [BIOMES.TROPICAL]: createColorSplineFromColorElevation(tropicalColors),
  [BIOMES.SIM_CITY]: createColorSplineFromColorElevation(simCityColors),
  [BIOMES.OCEAN]: createColorSplineFromColorElevation(oceanColors),
}
