uniform mat4 uProjectionMatrixInverse; // camera.projectionMatrixInverse
uniform mat4 uViewMatrixInverse; // camera.matrixWorld
uniform vec3 uCameraPosition;
uniform vec3 uCameraWorldDirection;
uniform float uTime;
uniform vec3 uSunPosition;

#define RAY_BETA vec3(5.5e-6, 13.0e-6, 22.4e-6) /* rayleigh, affects the color of the sky */
#define MIE_BETA vec3(21e-6) /* mie, affects the color of the blob around the sun */
#define AMBIENT_BETA vec3(0.0) /* ambient, affects the scattering color when there is no lighting from the sun */
#define ABSORPTION_BETA vec3(2.04e-5, 4.97e-5, 1.95e-6) /* what color gets absorbed by the atmosphere (Due to things like ozone) */
#define G 0.76 /* mie scattering direction, or how big the blob around the sun is */
// and the heights (how far to go up before the scattering has no effect)
#define HEIGHT_RAY 2e3 // 8e3 /* rayleigh height */
#define HEIGHT_MIE 5e2 //1.2e3 /* and mie, make this promenant to increase smogginess */
#define HEIGHT_ABSORPTION 30e3 /* at what height the absorption is at it's maximum */
#define ABSORPTION_FALLOFF 4e3 /* how much the absorption decreases the further away it gets from the maximum height */
// and the steps (more looks better, but is slower)
// the primary step has the most effect on looks
// and these on desktop


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

const int MAX_STEPS = 32;

// const vec3  SUN_COLOR = vec3(20.0, 19.0, 13.0);
const vec3  SUN_COLOR = vec3(10, 10, 10);
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

// struct Intersection {
//     vec3 distanceNear;
//     vec3 distancefar;
//     Ray begin;
//     Ray end;
// };


vec4 rayMarch(in Ray ray, in vec3 box, in vec3 boxPosition, in float maxDistance, in vec3 scene_color) {
   
    // light stuff
    // light_color = SUN_COLOR;
   
    float distanceTraveled = 0.0;
    vec3 color = vec3(0.0, 0.0, 0.0);
    Ray sun = Ray(uSunPosition, get_sun_direction(uSunPosition));
    vec3 accum = scene_color;
    
    float sun_phase = HG_phase(sun.direction, ray.direction, SUN_SCATTERING_ANISO)*3.0;
    // float signedDistance = sdBox(Translate( currentPosition, boxPosition), box);
    vec2 intersection = boxIntersection(Translate( ray.origin, boxPosition), ray.direction, box);
    float intersectionNear = intersection.x;
    float intersectionFar = intersection.y;
    bool objectInFront = maxDistance < intersectionNear;

    // no intersection
    if (intersection == vec2(-1.0)) return vec4(accum, 1.0);
    // terrain or other mesh in front of the sdf box
    if (objectInFront) return vec4(accum, 1.0);

    Ray begin = Ray(ray.origin + ray.direction * intersectionNear, ray.direction);
    // if we're inside the box, start at the input ray origin
    if (intersectionNear < 0.0) {
      begin = Ray(ray.origin, ray.direction);
    }
    Ray end = Ray(ray.origin + ray.direction * min(intersectionFar, maxDistance), ray.direction);

    float intersectionDistance = length(end.origin - begin.origin);
    float stepSize = intersectionDistance / float(MAX_STEPS);

    float fog = 0.0002 / float(MAX_STEPS);


  // SCATTERING THINGS
      float light_intensity = 10.0;
      vec3 beta_ray = RAY_BETA; 				// the amount rayleigh scattering scatters the colors (for earth: causes the blue atmosphere)
      vec3 beta_mie = MIE_BETA; 				// the amount mie scattering scatters colors
      vec3 beta_absorption = ABSORPTION_BETA;   	// how much air is absorbed
      vec3 beta_ambient = AMBIENT_BETA;			// the amount of scattering that always occurs, cna help make the back side of the atmosphere a bit brighter
      float g = G;					// the direction mie scatters the light in (like a cone). closer to -1 means more towards a single direction
      float height_ray = HEIGHT_RAY; 			// how high do you have to go before there is no rayleigh scattering?
      float height_mie = HEIGHT_MIE; 			// the same, but for mie
      float height_absorption = HEIGHT_ABSORPTION;	// the height at which the most absorption happens
      float absorption_falloff = ABSORPTION_FALLOFF;	// how fast the absorption falls off from the absorption height
      
      bool allow_mie = true;

      // these are the values we use to gather all the scattered light
      vec3 total_ray = vec3(0.0); // for rayleigh
      vec3 total_mie = vec3(0.0); // for mie

      // initialize the optical depth. This is used to calculate how much air was in the ray
      vec3 opt_i = vec3(0.0);

      // modulate the atmoshpere's density by this value
      float atmo_density = 1.0; // TODO make it an input
      float densityMultiplier = atmo_density;

      // also init the scale height, avoids some vec2's later on
      vec2 scale_height = vec2(height_ray * densityMultiplier, height_mie * densityMultiplier);
    

      // Calculate the Rayleigh and Mie phases.
      // This is the color that will be scattered for this ray
      // mu, mumu and gg are used quite a lot in the calculation, so to speed it up, precalculate them
      float mu = dot(ray.direction, sun.direction);
      float mumu = mu * mu;
      float gg = g * g;
      float phase_ray = 3.0 / (50.2654824574 /* (16 * pi) */) * (1.0 + mumu);
      float phase_mie = allow_mie ? 3.0 / (25.1327412287 /* (8 * pi) */) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg)) : 0.0;



// Offsetting the position used for querying occlusion along the world normal can be used to reduce shadow acne.
    vec3 shadowWorldNormal = inverseTransformDirection(sun.direction, viewMatrix );

    for(int i = 0; i < MAX_STEPS; ++i) {
      vec3 currentPosition = begin.origin + ray.direction * (stepSize * float(i));
      float height = currentPosition.y;

      // now calculate the density of the particles (both for rayleigh and mie)
      vec3 density = vec3(exp(-height / scale_height), 0.0);

      // and the absorption density. this is for ozone, which scales together with the rayleigh, 
      // but absorbs the most at a specific height, so use the sech function for a nice curve falloff for this height
      // clamp it to avoid it going out of bounds. This prevents weird black spheres on the night side
      float denom = (height_absorption - height) / absorption_falloff;
      density.z = (1.0 / (denom * denom + 1.0)) * density.x;
      // multiply it by the step size here
      // we are going to use the density later on as well
      density *= stepSize;
      // Add these densities to the optical depth, so that we know how many particles are on this ray.
      opt_i += density;

      // get light ray stuff
      vec2 lightRayIntersection = boxIntersection(Translate( currentPosition, boxPosition), sun.direction, box);
      float lightRayIntersectionNear = intersection.x;
      float lightRayIntersectionFar = intersection.y;
      bool objectInFront = maxDistance < intersectionNear;

      // // no intersection
      // if (intersection == vec2(-1.0)) return vec4(accum, 1.0);
      // // terrain or other mesh in front of the sdf box
      // if (objectInFront) return vec4(accum, 1.0);

      Ray lightBegin = Ray(currentPosition + sun.direction * lightRayIntersectionNear, sun.direction);
      // if we're inside the box, start at the input ray origin
      if (lightRayIntersectionNear < 0.0) {
        lightBegin = Ray(currentPosition, sun.direction);
      }
      // TODO make this end with
      Ray lightEnd = Ray(currentPosition + sun.direction * lightRayIntersectionFar, sun.direction);

      float lightIntersectionDistance = length(end.origin - begin.origin);
      float lightStepSize = intersectionDistance / float(MAX_STEPS);
      int lightSteps = int(lightIntersectionDistance / lightStepSize);
      // and the optical depth of this ray
      vec3 opt_l = vec3(0.0);

      // shadow stuff
      vec4 shadowWorldPosition = vec4(currentPosition, 1.0) + vec4( shadowWorldNormal * uDirectionalLightShadow.normalBias, 0. ); //+ vec4(offset, 0.); // <-- see offset
      vec4 directionalShadowCoord = uDirectionalShadowMatrix * shadowWorldPosition;

      // vec4 shadowCoord = uDirectionalShadowMatrix * vec4(currentPosition, 1.0);
      directionalShadowCoord.xyz /= directionalShadowCoord.w;

      float shadowDepth = texture(uDirectionalShadowMap, directionalShadowCoord.xy).r;
      // shadowDepth = unpackRGBAToDepth( texture2D( uDirectionalShadowMap, shadowCoord.xy ) );
      shadowDepth = unpackRGBAToDepth( texture2D( uDirectionalShadowMap, directionalShadowCoord.xy ) );
      
      // we're in shadow
      if (shadowDepth < directionalShadowCoord.z) {
        lightSteps = 0;
        continue;
      }

      //  // now sample the light ray
      // // this is similar to what we did before
      for (int l = 0; l < lightSteps; ++l) {
          vec3 currentPositionOnLightRay = lightBegin.origin + lightBegin.direction * (lightStepSize * float(l));
          float lightHeight = currentPositionOnLightRay.y;

          // calculate the particle density, and add it
          // this is a bit verbose
          // first, set the density for ray and mie
          vec3 density_l = vec3(exp(-lightHeight / scale_height), 0.0);

          // then, the absorption
          float denom = (height_absorption - lightHeight) / absorption_falloff;
          density_l.z = (1.0 / (denom * denom + 1.0)) * density_l.x;
          
          // multiply the density by the step size
          density_l *= lightStepSize;
          
          // and add it to the total optical depth
          opt_l += density_l;
      }

       // Now we need to calculate the attenuation
      // this is essentially how much light reaches the current sample point due to scattering
      vec3 attn = exp(-beta_ray * (opt_i.x + opt_l.x) - beta_mie * (opt_i.y + opt_l.y) - beta_absorption * (opt_i.z + opt_l.z));

      // accumulate the scattered light (how much will be scattered towards the camera)
      total_ray += density.x * attn;
      total_mie += density.y * attn;


      // float height_factor = clamp(remap(height, 400.0, 10000.0, 1.0, 0.0), 0.0, 1.0);
      // height_factor = easeInExpo(height_factor);
      
      // // shadow stuff
      // vec4 shadowWorldPosition = vec4(currentPosition, 1.0) + vec4( shadowWorldNormal * uDirectionalLightShadow.normalBias, 0. ); //+ vec4(offset, 0.); // <-- see offset
      // vec4 directionalShadowCoord = uDirectionalShadowMatrix * shadowWorldPosition;

      // // vec4 shadowCoord = uDirectionalShadowMatrix * vec4(currentPosition, 1.0);
      // directionalShadowCoord.xyz /= directionalShadowCoord.w;

      // float shadowDepth = texture(uDirectionalShadowMap, directionalShadowCoord.xy).r;
      // // shadowDepth = unpackRGBAToDepth( texture2D( uDirectionalShadowMap, shadowCoord.xy ) );
      // shadowDepth = unpackRGBAToDepth( texture2D( uDirectionalShadowMap, directionalShadowCoord.xy ) );

      // float dianceToSun = length(uSunPosition - currentPosition);

      // // only accumulate if we're in the atmosphere
  
      // vec3 sky = SKY_COLOR * (height_factor * 0.2) * stepSize;
      // vec3 sun = SUN_COLOR * sun_phase * (height_factor * 0.5 )  * stepSize;
      
      // // accum += sky * fog;
      // // accum += sun * fog;

      //           // Point is in shadow
      //   if (shadowDepth < directionalShadowCoord.z) {
      //       // accum += SHADOW_COLOR * vec3(shadowDepth);
      //   } else {
      //       accum += sky * fog;
      //       accum += sun * fog;
      //   }
      // accum += SHADOW_COLOR * vec3(shadowDepth);
    }

    // calculate how much light can pass through the atmosphere
    vec3 opacity = exp(-(beta_mie * opt_i.y + beta_ray * opt_i.x + beta_absorption * opt_i.z));

    accum = vec3(
        	phase_ray * beta_ray * total_ray * SUN_COLOR// rayleigh color
       		+ phase_mie * beta_mie * total_mie * SUN_COLOR// mie
            + opt_i.x * beta_ambient // and ambient
    ) * light_intensity + scene_color * opacity; // now make sure the background is rendered correctly

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