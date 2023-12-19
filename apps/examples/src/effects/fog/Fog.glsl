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


struct Ray {
    vec3 origin;
    vec3 direction;
};

vec3 Translate(in vec3 p, in vec3 t) {
    return p - t;
}

const int MAX_STEPS = 256;

// Ray marching constants:
const vec3 GRADIENT_STEP = vec3(0.01, 0.0, 0.0);
const float MAX_TRACE_DISTANCE = 100000.0;
const float MIN_HIT_DISTANCE = 0.1;
const float DISTANCE_STEP = 10.0;
const int MAX_INTERSECTIONS = 1;

const float FOG_HEIGHT_FALLOFF = 0.0000001;
const float FOG_GLOBAL_DENSITY = 0.0000001;

const int FOG_TRACE_STEPS = 32;

const float FOG_NOISE_SCALE = 0.004;
const float FOG_NOISE_SPEED = 0.07;

const float CLOUDS_DENSITY = 0.0008;
const float CLOUDS_MAX_DENSITY_HEIGHT  = 500.0;
const float CLOUDS_ZERO_DENSITY_HEIGHT = 1000.0;

const vec3  SUN_COLOR = vec3(20.0, 19.0, 13.0);
const vec3  SKY_COLOR = vec3(50.0, 100.0, 200.0);
const float SUN_SCATTERING_ANISO = 0.07;

float b = 0.01;
float a = 0.00001;

vec3 applyFog( in vec3  col, // color of pixel
               in float t,
               in float density)  // distance to point
{
    float fogAmount = 1.0 - exp(-t*density);
    vec3  fogColor  = vec3(0.5,0.6,0.7);
    return mix( col, fogColor, fogAmount );
}

// float get_shadow(in vec3 wpos)
// {
//     vec3 dummy;
// 	return terrain_intersect(wpos + vec3(0.0, 0.1, 0.0), get_sun_direction(), 4.0, 40, 0, dummy) ? 0.0 : 1.0;
// }

float get_fog_density(in vec3 pos)
{
    vec3 coord = pos*FOG_NOISE_SCALE;
    // coord.x += iTime * FOG_NOISE_SPEED;
    
    // float noise = texture(iChannel1, coord).x;
    float noise = 0.5;
    float exp_fog = exp(-pos.y*FOG_HEIGHT_FALLOFF)*FOG_GLOBAL_DENSITY*noise;
    
    float cloud_fog = CLOUDS_DENSITY;
    float k = clamp((pos.y - CLOUDS_MAX_DENSITY_HEIGHT) / (CLOUDS_ZERO_DENSITY_HEIGHT - CLOUDS_MAX_DENSITY_HEIGHT), 0.0, 1.0);
    cloud_fog *= 1.0 - k;

    cloud_fog *= noise;
    
    return exp_fog + cloud_fog;
}


// Henyey-Greenstein phase function
float HG_phase(in vec3 L, in vec3 V, in float aniso)
{
    float cosT = dot(L,-V);
    float g = aniso;
    return (1.0-g*g) / (4.0*PI*pow(1.0 + g*g - 2.0*g*cosT, 3.0/2.0));
}

vec3 get_sun_direction(in vec3 pos)
{
    //return SUN_DIRECTION;

    // float angle = iTime/16.0;
    
    float angle = 1.9;
    
    vec3 dir = vec3(pos - vec3(6244.923261707597,
  6953.7247328594185,
  6263.770656081942));
    dir = normalize(dir);
    
    // dir = vec3(0.0, 1.0, 0.0);
    
    return dir;
}

// vec3 apply_volumetric_fog(in vec3 eye, in vec3 pos, in vec3 scene_color, in float noise)
// {
//     vec3 dir = eye - pos;
//     vec3 V = normalize(dir);
//     vec3 L = get_sun_direction();
    
//     vec3 accum = scene_color;
    
//     float sun_phase = HG_phase(L, V, SUN_SCATTERING_ANISO)*3.0;
    
//     float step = length(dir) / float(FOG_TRACE_STEPS) / 1.0;
    
//     float jitter = noise*2.5;
    
//     for(int i=0; i<FOG_TRACE_STEPS; ++i)
//     {
//         float k = float(i)/float(FOG_TRACE_STEPS-1);
//         k += jitter;
        
//         vec3  pi = pos + dir * k;
//         // float s = get_shadow(pi);
//         float s = 1.0;
//         float f = get_fog_density(pi);
        
//         float T = exp(-f*step);
        
//         vec3 sky = SKY_COLOR * step;
//         vec3 sun = SUN_COLOR * sun_phase * s * step;
        
//         accum = accum*T;
//         accum += sky * f;
//         accum += sun * f;
//     }
//     return accum;
// }

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
    // outNormal = (tN>0.0) ? step(vec3(tN),t1)) : // ro ouside the box
    //                        step(t2,vec3(tF)));  // ro inside the box
    // outNormal *= -sign(rd);
    return vec2( tN, tF );
}

float remap( in float value, in float x1, in float y1, in float x2, in float y2) {
  return ((value - x1) * (y2 - x2)) / (y1 - x1) + x2;
}

vec4 rayMarch(in Ray ray, in vec3 box, in vec3 boxPosition, in float maxDistance, in vec3 scene_color) {
    float distanceTraveled = 0.0;
    vec3 color = vec3(0.0, 0.0, 0.0);
    int intersections = 0;


    vec3 sunDir = get_sun_direction(ray.origin);
    vec3 accum = scene_color;
    
    float sun_phase = HG_phase(sunDir, ray.direction, SUN_SCATTERING_ANISO)*3.0;
    // float signedDistance = sdBox(Translate( currentPosition, boxPosition), box);
    float distanceToTravel = min(maxDistance, 12000.0)/ float(MAX_STEPS) / 1.0;

    vec2 intersection = boxIntersection(Translate( ray.origin, boxPosition), ray.direction, box);
    if (intersection == vec2(-1.0)) return vec4(accum, 1.0);
    if (maxDistance < intersection.x) return vec4(accum, 1.0);

    for(int i=0; i<MAX_STEPS; ++i){
        vec3 currentPosition = ray.origin + ray.direction * distanceTraveled;
        vec2 signedDistance = boxIntersection(Translate( currentPosition, boxPosition), ray.direction, box);
        // float density = 1.0 - exp(currentPosition.y / 8000.0);
        if (signedDistance.y > 0.0) {
          // float fog = get_fog_density(currentPosition);
          //   // vec3 normal = calculateNormal(currentPosition);
          //   // color += vec3(0.0, 0.0, 0.1); //shadeSurface(sceneSurface, currentPosition, ray, normal);
          //   // color = applyFog(color, distanceTraveled, heightFallOff);
          //   // color += vec3(0.0, 0.0, currentPosition.y / 1000000.0); 
            distanceTraveled += distanceToTravel;
          //   intersections++;

          //   float T = exp(-fog*distanceToTravel);
          // float fogDensity = 0.000000001;
          // float fogDepth = maxDistance;
          // float heightFactor = 0.05;
          // float fogFactor = heightFactor * exp(-currentPosition.y * fogDensity) * (
          //     1.0 - exp(-fogDepth * currentPosition.y * fogDensity)) / ray.direction.y;
          // fogFactor = saturate(fogFactor);
          // float fogHeightStart = 0.0;
          // float fogHeightEnd = 5000.0;

          // float height_factor = clamp((currentPosition.y - fogHeightStart) / (fogHeightEnd - fogHeightStart), 0.0, 1.0);
  float height_factor = clamp(remap(currentPosition.y, 0.0, 8000.0, 0.5, 0.0), 0.0, 1.0);
          
            // only accumulate if we're in the atmosphere
            float fog = 0.0000005;
        
            vec3 sky = SKY_COLOR * (height_factor * 0.8) * distanceToTravel;
            vec3 sun = SUN_COLOR * sun_phase * height_factor  * distanceToTravel;
            
          //   accum = accum*T;
            accum += sky * fog;
            accum += sun * fog;

        }
       

        if (distanceTraveled > maxDistance || intersections > MAX_INTERSECTIONS) {
          break;
        }


        distanceTraveled += distanceToTravel;
    }

    return vec4(accum, 1.0);
}

// float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
//     float depth = start;
//     vec3 box = vec3(1000., 1000., 1000.);
//     for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
//         float dist = sdBox(eye + depth * marchingDirection, box);
//         if (dist < FUDGE_ZONE) {
// 			return depth;
//         }
//         depth += dist;
//         if (depth >= end) {
//             return end;
//         }
//     }
//     return end;
// }

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
  // float sdBoxDepth = sdBox(pos, vec3(10000., 400., 10000.));

  float atmo_radius = 1000.0;
  Ray ray = Ray(rayOrigin, rayDirection);
  vec4 color = rayMarch(ray, vec3(5000., 5000., 5000.) - vec3(10.), vec3(0.0, 4500.0, 0.0), sceneDepth, inputColor.xyz);

  // float dist = shortestDistanceToSurface(rayOrigin, -rayDirection, MIN_DIST, MAX_DIST);

  // if (boop < 0.) {
  //   addColor = addColor + vec4(0.0, 1.0, 0.0, 1.0);
  //   // addColor = applyFog(inputColor, sceneDepth, rayOrigin, rayDirection);
  // } else {
  //   // addColor = vec3(1., 0., 0.);
  // }

  outputColor = vec4(color.xyz, 1.0);
}