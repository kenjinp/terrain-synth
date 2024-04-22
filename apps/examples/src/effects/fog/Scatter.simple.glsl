

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