import { SourceHttp } from "@chunkd/source-http"
import { CogTiff } from "@cogeotiff/core"

const blah =
  "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_2022_2697-1136/swissalti3d_2022_2697-1136_2_2056_5728.tif"

export const generateGeoData = async (url: string) => {
  const source = new SourceHttp(blah)

  //   export interface Source {
  //     /** Where the source is located */
  //     url: URL;
  //     /** Optional metadata about the source including the size in bytes of the file */
  //     metadata?: {
  //         /** Number of bytes in the file if known */
  //         size?: number;
  //     };
  //     /** Fetch bytes from a source */
  //     fetch(offset: number, length?: number): Promise<ArrayBuffer>;
  //     /** Optionally close the source, useful for sources that have open connections of file descriptors */
  //     close?(): Promise<void>;
  // }

  const tiff = await CogTiff.create(source)
  console.log({ tiff })

  /** Load a specific tile from a specific image */
  // const tile = await tiff.images[5].getTile(2, 2)

  /** Load the 5th image in the Tiff */
  const img = tiff.images[1]
  // if (img.isTiled()) {
  //   /** Load tile x:10 y:10 */
  //   const tile = await img.getTile(10, 10)
  //   tile.mimeType // image/jpeg
  //   tile.bytes // Raw image buffer
  //   console.log({ tile })
  // }

  /** Get the origin point of the tiff */
  const origin = img.origin
  /** Bounding box of the tiff */
  const bbox = img.bbox

  console.log({ img })
  return img
}
