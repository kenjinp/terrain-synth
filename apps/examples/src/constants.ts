declare const __COMMIT_INFO__: {
  shortHash: string
  hash: string
  subject: string
  sanitizedSubject: string
  body: string
  authoredOn: string
  committedOn: string
  author: {
    name: string
    email: string
  }
  committer: {
    name: string
    email: string
  }
  notes: string
  branch: string
  tags: string[]
}

declare const __BUILD_INFO__: {
  buildTime: number
}

export const COMMIT_INFO = __COMMIT_INFO__
export const BUILD_INFO = __BUILD_INFO__
