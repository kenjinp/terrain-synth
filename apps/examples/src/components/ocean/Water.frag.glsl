
uniform sampler2D u_sceneTexture; // The Scene texture
uniform sampler2D u_depthTexture; // The WebGL depth texture
uniform sampler2D u_normalTexture; // Water normal texture
uniform vec2 u_winRes; // Viewport resolution
uniform float u_near;
uniform float u_far;
uniform float u_time;
uniform vec3 u_lightPos;

// Controllable uniforms
uniform float u_absorbStrength;
uniform float u_shine;
uniform float u_diffuse;
uniform bool u_stepped;

varying vec4 v_local_position; // The local position of the geometry
varying vec4 v_viewPosition; // The view position of the geometry
varying vec4 v_worldPosition; // The world position of the geometry
varying vec3 v_normal; // The normals of the geometry
varying vec2 v_uv; // The uv of the geometry

float random(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 hsl2rgb(vec3 c){
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0);
    return c.z + c.y * (rgb-0.5)*(1.0-abs(2.0*c.z-1.0));
}

vec3 rgb2hsl( in vec3 c ){
  float h = 0.0;
	float s = 0.0;
	float l = 0.0;
	float r = c.r;
	float g = c.g;
	float b = c.b;
	float cMin = min( r, min( g, b ) );
	float cMax = max( r, max( g, b ) );

	l = ( cMax + cMin ) / 2.0;
	if ( cMax > cMin ) {
		float cDelta = cMax - cMin;
        
        //s = l < .05 ? cDelta / ( cMax + cMin ) : cDelta / ( 2.0 - ( cMax + cMin ) ); Original
		s = l < .0 ? cDelta / ( cMax + cMin ) : cDelta / ( 2.0 - ( cMax + cMin ) );
        
		if ( r == cMax ) {
			h = ( g - b ) / cDelta;
		} else if ( g == cMax ) {
			h = 2.0 + ( b - r ) / cDelta;
		} else {
			h = 4.0 + ( r - g ) / cDelta;
		}

		if ( h < 0.0) {
			h += 6.0;
		}
		h = h / 6.0;
	}
	return vec3( h, s, l );
}

vec4 fromLinear(vec4 linearRGB) {
    bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055)*pow(linearRGB.rgb, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = linearRGB.rgb * vec3(12.92);
    return vec4(mix(higher, lower, cutoff), linearRGB.a);
}

float A_logDepthBufFC () {
  float logDepthBufFC = 2.0 / ( log( u_far + 1.0 ) / log(2.0) );
  return logDepthBufFC;
}

// float readDepth( sampler2D depthSampler, vec2 coord ) {
//     float fragCoordZ = texture2D( depthSampler, coord ).x;
//     float v_depth = pow(2.0, fragCoordZ / (A_logDepthBufFC() * 0.5));
//     // float z_view = v_depth - 1.0;
//     return v_depth;
//     // return u_far * u_near / ((u_near-u_far) * fragCoordZ + u_far);
// }

vec3 posterize(vec3 inputColor){
    float gamma = 0.8;
    float numColors = 20.0;

    vec3 c = inputColor.rgb;
    c = pow(c, vec3(gamma, gamma, gamma));
    c = c * numColors;
    c = floor(c);
    c = c / numColors;
    c = pow(c, vec3(1.0/gamma));

    return vec3(c);
}

struct Geometry {
	vec3 pos;
	vec3 worldPos;
    vec3 viewPos;
	vec3 viewDir;
	vec3 worldViewDir;
	vec3 normal;
};

float readDepth( float z ) {
  return perspectiveDepthToViewZ( z, u_near, u_far );
}


void main() {

    // Initialise
    vec3 finalColor;
    vec2 uv = v_uv;
    vec2 screenSpace = gl_FragCoord.xy / u_winRes.xy;
    vec3 normalTextureR = texture2D(u_normalTexture, 5.0 * uv + (u_time * 0.1)).rgb;
    vec3 normalTextureL = texture2D(u_normalTexture, 5.0 * uv - u_time).rgb;
    vec4 sceneTexture = fromLinear(texture2D(u_sceneTexture, screenSpace));


    // float d = readDepth(texture2D(u_depthTexture, screenSpace).x);
    // float v_depth = pow(2.0, d / (A_logDepthBufFC() * 0.5));
    // float z_view = v_depth - 1.0;
    
    // straight depth
    float z = texture2D(u_depthTexture, screenSpace).x;
    float depthZ = (exp2(z / (A_logDepthBufFC() * 0.5)) - 1.0);

    // vec3 posWS = _ScreenToWorld(vec3(uv, z));

    // Water geometry struct
    Geometry geo;
    geo.pos = v_local_position.xyz;
    geo.worldPos = v_worldPosition.xyz;
    geo.viewPos = -v_viewPosition.xyz;
    geo.worldViewDir = geo.worldPos - cameraPosition;
    geo.viewDir = normalize(v_viewPosition.xyz);
    geo.normal = v_normal;
    
    vec3 lightDir = normalize(u_lightPos - geo.worldPos);

    // // View-space water depth
    // distToWater = gl_FragCoord.z / gl_FragCoord.w;
    // float viewWaterDepth = sceneDepth - distToWater;

    // World-space water depth
    float sceneDepth = depthZ; //readDepth(u_depthTexture, screenSpace);
    float distToWater = geo.viewPos.z;
    vec3 wDirDivide = geo.worldViewDir / distToWater;
    vec3 wDirMultiply = wDirDivide * sceneDepth;
    vec3 wDirAdd = wDirMultiply + cameraPosition;
    vec3 wDirSubtract = geo.worldPos - wDirAdd;
    float worldWaterDepth = length(wDirSubtract.y);

    // Water Color
    vec3 waterColor = vec3(184.95/360.0, 0.5243, 0.5627); waterColor = hsl2rgb(waterColor);
    vec3 absorbColor = vec3(1.0) - waterColor;
    float absorbVal = 1.0 - exp(-0.0005 * depthZ);
    vec3 sceneColor = sceneTexture.rgb;
    vec3 subtractiveColor = absorbColor * absorbVal;
    vec3 underWaterColor = sceneColor - subtractiveColor;

    finalColor = mix(waterColor, underWaterColor, 0.5) * saturate(1.0 - absorbVal);
    // finalColor = vec3(sceneDepth);

    // Lights
    // float NdotL= dot(geo.normal, lightDir);
    // float lightIntensity = smoothstep(0.0, 0.01, NdotL);
    // vec3 directionalLightColor = directionalLights[0].color * lightIntensity;
    

    // vec3 halfVector = normalize(lightDir + geo.viewPos);
    // float NdotH = dot(geo.normal, halfVector);

    // float specularIntensity = pow(NdotH * lightIntensity, 1000.0 / u_shine);
    // float specularIntensitySmooth = smoothstep(0.0, 0.08, specularIntensity);
    // vec3 specular = specularIntensitySmooth * directionalLights[0].color;
    // vec3 outLightColor = directionalLightColor + specular + ambientLightColor;

    // finalColor *= outLightColor;

    // Final Out

    // if (u_stepped) { finalColor = floor(finalColor * 20.0) / 20.0; }
    csm_FragColor = vec4(finalColor, 1.0);

    // Debug
    // gl_FragColor = vec4(outLightColor, 1.0);
}