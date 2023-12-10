uniform mat4 uProjectionMatrixInverse; // camera.projectionMatrixInverse
uniform mat4 uViewMatrixInverse; // camera.matrixWorld
uniform vec3 uCameraPosition;
uniform vec3 uCameraWorldDirection;

#define saturate(a) clamp( a, 0.0, 1.0 )

const float MIN_DIST = 0.0;
const float MAX_DIST = 100000.0;
const float FUDGE_ZONE = 1e-6;
const int MAX_MARCHING_STEPS = 1000;

vec3 _ScreenToWorld(vec3 posS) {
  vec2 uv = posS.xy;
  float z = posS.z;
  float nearZ = 0.01;
  float farZ = cameraFar;
  float depth = pow(2.0, z * log2(farZ + 1.0)) - 1.0;
  vec3 direction = (uProjectionMatrixInverse * vec4(vUv * 2.0 - 1.0, 0.0, 1.0)).xyz;
  direction = (uViewMatrixInverse * vec4(direction, 0.0)).xyz;
  direction = normalize(direction);
  direction /= dot(direction, uCameraWorldDirection);
  return uCameraPosition + direction * depth;
}

float readDepth( float z ) {
  return perspectiveDepthToViewZ( z, cameraNear, cameraFar );
}

float A_logDepthBufFC () {
  float logDepthBufFC = 2.0 / ( log( cameraFar + 1.0 ) / log(2.0) );
  return logDepthBufFC;
}




vec4 applyFog( in vec4  col,  // color of pixel
               in float t,    // distnace to point
               in vec3  ro,   // camera position
               in vec3  rd )  // camera to point vector
{
    float b = 0.0001;
    float a = 0.0001;
    float fogAmount = (a/b) * exp(-ro.y*b) * (1.0-exp(-t*rd.y*b))/rd.y;
    vec4  fogColor  = vec4(0.5,0.6,0.7, 1.0);
    return mix( col, fogColor, fogAmount );
}

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    vec3 box = vec3(1000., 1000., 1000.);
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sdBox(eye + depth * marchingDirection, box);
        if (dist < FUDGE_ZONE) {
			return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}

void mainImage(const in vec4 inputColor, const in vec2 uv, const in float depth, out vec4 outputColor) {
  float depthValue = getViewZ(depth);
  float d = readDepth(texture2D(depthBuffer, uv).x);
  float v_depth = pow(2.0, d / (A_logDepthBufFC() * 0.5));
  float z_view = v_depth - 1.0;
  
  // straight depth
  float z = texture2D(depthBuffer, uv).x;
  float depthZ = (exp2(z / (A_logDepthBufFC() * 0.5)) - 1.0);

  vec3 posWS = _ScreenToWorld(vec3(uv, z));
  vec3 rayOrigin = uCameraPosition;
  vec3 rayDirection = normalize(posWS - uCameraPosition);
  float sceneDepth = length(posWS.xyz - uCameraPosition);
  vec4 addColor = inputColor;
  vec3 pos = rayOrigin - vec3(0.0);
  float sdBoxDepth = sdBox(pos, vec3(10000., 1000., 10000.));

  float atmo_radius = 1000.0;
  vec3 dir = rayDirection;
  vec3 start = rayOrigin;
  float a = dot(dir, dir);
  float b = 2.0 * dot(dir, start);
  float c = dot(start, start) - (atmo_radius * atmo_radius);
  float blah = (b * b) - 4.0 * a * c;

  float boop = sdBox(posWS, vec3(1000., 1000., 1000.));
  float dist = shortestDistanceToSurface(rayOrigin, -rayDirection, MIN_DIST, MAX_DIST);

  if (blah > 0.) {
    // addColor = vec3(0., 1., 0.);
    addColor = applyFog(inputColor, sceneDepth, rayOrigin, rayDirection);
  } else {
    // addColor = vec3(1., 0., 0.);
  }

  outputColor = vec4(addColor);
}