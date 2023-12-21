declare module "*.glsl" {
  const value: string
  export default value
}

declare module "glslify" {
  const value: function(TemplateStringsArray): string
  export default value
}
