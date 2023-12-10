import { dirname } from "path"
import { fileURLToPath } from "url"

const __filenameNew = fileURLToPath(import.meta.url)
export const __dirname = dirname(__filenameNew)
