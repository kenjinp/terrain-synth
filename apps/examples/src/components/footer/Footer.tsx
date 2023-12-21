import { BUILD_INFO, COMMIT_INFO } from "../../constants"
import "./Footer.css"

export const Footer: React.FC<React.PropsWithChildren> = ({ children }) => {
  return (
    <footer className="main">
      <div>
        version{" "}
        <a
          title="commit hash"
          href={`https://github.com/kenjinp/terrain-synth/commit/${COMMIT_INFO.hash}`}
        >
          {COMMIT_INFO.shortHash}
        </a>{" "}
        <span title="build date">
          {new Date(BUILD_INFO.buildTime).toLocaleDateString()} WIP
        </span>
      </div>
      {children}
      <div>
        <a href="https://ko-fi.com/kennywtf">Support this project</a>
      </div>
    </footer>
  )
}
