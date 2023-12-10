// from https://github.com/seymen/git-last-commit
import { execSync } from "child_process"
const splitCharacter = "<##>"

const executeCommand = (command: any, options: any) => {
  let dst = __dirname

  if (!!options && options.dst) {
    dst = options.dst
  }

  const stdout = execSync(command, { cwd: dst }).toString()
  if (stdout === "") {
    return
  }

  return stdout
}

const prettyFormat = [
  "%h",
  "%H",
  "%s",
  "%f",
  "%b",
  "%at",
  "%ct",
  "%an",
  "%ae",
  "%cn",
  "%ce",
  "%N",
  "",
]

const getCommandString = (splitCharacter: string) =>
  'git log -1 --pretty=format:"' +
  prettyFormat.join(splitCharacter) +
  '"' +
  " && git rev-parse --abbrev-ref HEAD" +
  " && git tag --contains HEAD"

export const getLastCommit = (options: any) => {
  const command = getCommandString(splitCharacter)

  const res = executeCommand(command, options) as string

  const a = res.split(splitCharacter)

  // e.g. master\n or master\nv1.1\n or master\nv1.1\nv1.2\n
  const branchAndTags = a[a.length - 1].split("\n").filter(n => n)
  const branch = branchAndTags[0]
  const tags = branchAndTags.slice(1)

  return {
    shortHash: a[0],
    hash: a[1],
    subject: a[2],
    sanitizedSubject: a[3],
    body: a[4],
    authoredOn: a[5],
    committedOn: a[6],
    author: {
      name: a[7],
      email: a[8],
    },
    committer: {
      name: a[9],
      email: a[10],
    },
    notes: a[11],
    branch,
    tags,
  }
}
