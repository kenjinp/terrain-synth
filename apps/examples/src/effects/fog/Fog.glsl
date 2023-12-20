uniform mat4 uProjectionMatrixInverse; // camera.projectionMatrixInverse
uniform mat4 uViewMatrixInverse; // camera.matrixWorld
uniform vec3 uCameraPosition;
uniform vec3 uCameraWorldDirection;
uniform float uTime;
uniform vec3 uSunPosition;


#include "./Noise.glsl";

// https://github.com/mrdoob/three.js/blob/fe312e19c2d8fa4219d035f0b83bc13a46fb1927/src/renderers/shaders/ShaderChunk/packing.glsl.js#L24

#define saturate(a) clamp( a, 0.0, 1.0 )

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

struct Ray {
    vec3 origin;
    vec3 direction;
};

vec3 Translate(in vec3 p, in vec3 t) {
    return p - t;
}

const int MAX_STEPS = 64;

const vec3  SUN_COLOR = vec3(20.0, 19.0, 13.0);
const vec3  SKY_COLOR = vec3(50.0, 100.0, 200.0);
const vec3 SHADOW_COLOR = vec3(200.0, 0.0, 0.0);
const float SUN_SCATTERING_ANISO = 0.07;


uniform sampler2D uDirectionalShadowMap;
uniform mat4 uDirectionalShadowMatrix;

struct DirectionalLightShadow {
  float bias;
  float normalBias;
  float radius;
  vec2 mapSize;
};

uniform DirectionalLightShadow uDirectionalLightShadow;

// Henyey-Greenstein phase function
float HG_phase(in vec3 L, in vec3 V, in float aniso)
{
    float cosT = dot(L,-V);
    float g = aniso;
    return (1.0-g*g) / (4.0*PI*pow(1.0 + g*g - 2.0*g*cosT, 3.0/2.0));
}

vec3 get_sun_direction(in vec3 pos)
{
    float angle = 1.9;
    // Hardcoded to match sun in three.js
    // should pass this from the tsx component
    vec3 dir = vec3(pos - uSunPosition);
    dir = normalize(dir);
    
    return dir;
}

	// vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	// vec4 shadowWorldPosition;

vec2 boxIntersection( in vec3 ro, in vec3 rd, vec3 boxSize) 
{
    vec3 m = 1.0/rd; // can precompute if traversing a set of aligned boxes
    vec3 n = m*ro;   // can precompute if traversing a set of aligned boxes
    vec3 k = abs(m)*boxSize;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    if( tN>tF || tF<0.0) return vec2(-1.0); // no intersection
    return vec2( tN, tF );
}

float remap( in float value, in float x1, in float y1, in float x2, in float y2) {
  return ((value - x1) * (y2 - x2)) / (y1 - x1) + x2;
}

float easeOutExpo(in float x) {
  return x == 1. ? 1. : 1. - pow(2., -10. * x);
}

float easeInExpo(in float x) {
  return x == 0. ? 0. : pow(2., 10. * x - 10.);
}

vec4 rayMarch(in Ray ray, in vec3 box, in vec3 boxPosition, in float maxDistance, in vec3 scene_color) {
    float distanceTraveled = 0.0;
    vec3 color = vec3(0.0, 0.0, 0.0);
    vec3 sunDir = get_sun_direction(ray.origin);
    vec3 accum = scene_color;
    
    float sun_phase = HG_phase(sunDir, ray.direction, SUN_SCATTERING_ANISO)*3.0;
    // float signedDistance = sdBox(Translate( currentPosition, boxPosition), box);
    vec2 intersection = boxIntersection(Translate( ray.origin, boxPosition), ray.direction, box);
    float intersectionNear = intersection.x;
    float intersectionFar = intersection.y;
    // no intersection
    if (intersection == vec2(-1.0)) return vec4(accum, 1.0);
    // terrain or other mesh in front of the sdf box
    if (maxDistance < intersectionNear) return vec4(accum, 1.0);

    Ray begin = Ray(ray.origin + ray.direction * intersectionNear, ray.direction);
    // if we're inside the box, start at the input ray origin
    if (intersectionNear < 0.0) {
      begin = Ray(ray.origin, ray.direction);
    }
    Ray end = Ray(ray.origin + ray.direction * min(intersectionFar, maxDistance), ray.direction);

    float intersectionDistance = length(end.origin - begin.origin);
    float distancePerStep = intersectionDistance / float(MAX_STEPS);

    float fog = 0.0002 / float(MAX_STEPS);

// Offsetting the position used for querying occlusion along the world normal can be used to reduce shadow acne.
    vec3 shadowWorldNormal = inverseTransformDirection(sunDir, viewMatrix );

    for(int i = 0; i < MAX_STEPS; ++i) {
      vec3 currentPosition = begin.origin + ray.direction * (distancePerStep * float(i));
      
      float height = currentPosition.y;
      float height_factor = clamp(remap(height, 400.0, 10000.0, 1.0, 0.0), 0.0, 1.0);
      height_factor = easeInExpo(height_factor);

      // shadow stuff
      vec4 shadowWorldPosition = vec4(currentPosition, 1.0) + vec4( shadowWorldNormal * uDirectionalLightShadow.normalBias, 0. ); //+ vec4(offset, 0.); // <-- see offset
      vec4 directionalShadowCoord = uDirectionalShadowMatrix * shadowWorldPosition;

      // vec4 shadowCoord = uDirectionalShadowMatrix * vec4(currentPosition, 1.0);
      directionalShadowCoord.xyz /= directionalShadowCoord.w;

      float shadowDepth = texture(uDirectionalShadowMap, directionalShadowCoord.xy).r;
      // shadowDepth = unpackRGBAToDepth( texture2D( uDirectionalShadowMap, shadowCoord.xy ) );
      shadowDepth = unpackRGBAToDepth( texture2D( uDirectionalShadowMap, directionalShadowCoord.xy ) );

      float dianceToSun = length(uSunPosition - currentPosition);

      // only accumulate if we're in the atmosphere
  
      vec3 sky = SKY_COLOR * (height_factor * 0.2) * distancePerStep;
      vec3 sun = SUN_COLOR * sun_phase * (height_factor * 0.5 )  * distancePerStep;
      
      // accum += sky * fog;
      // accum += sun * fog;

                // Point is in shadow
        if (shadowDepth < directionalShadowCoord.z) {
            // accum += SHADOW_COLOR * vec3(shadowDepth);
        } else {
            accum += sky * fog;
            accum += sun * fog;
        }
      // accum += SHADOW_COLOR * vec3(shadowDepth);
    }

    return vec4(accum, 1.0);
}

void mainImage(const in vec4 inputColor, const in vec2 uv, const in float depth, out vec4 outputColor) {
  float depthValue = getViewZ(depth);
  float d = readDepth(texture2D(depthBuffer, uv).x);
  float v_depth = pow(2.0, d / (A_logDepthBufFC() * 0.5));
  float z_view = v_depth - 1.0;
  
  float z = texture2D(depthBuffer, uv).x;
  float depthZ = (exp2(z / (A_logDepthBufFC() * 0.5)) - 1.0);
  vec3 posWS = _ScreenToWorld(vec3(uv, z));
  
  vec3 rayOrigin = uCameraPosition;
  vec3 rayDirection = normalize(posWS - uCameraPosition);

  float sceneDepth = length(posWS.xyz - uCameraPosition);

  Ray ray = Ray(rayOrigin, rayDirection);

  vec4 color = rayMarch(ray, vec3(5000., 5000., 5000.) - vec3(1.0), vec3(0.0, 4700.0, 0.0), sceneDepth, inputColor.xyz);

  outputColor = vec4(color.xyz, 1.0);
}