# Terrain Synth

Example-based Terrain Generation powered by real-earth datasets.

<img src="https://github.com/kenjinp/terrain-synth/blob/main/media/example.png?raw=true" alt="image of the terrain renderer" />

This is a platform for me to hack around on various machine-learning content generation / manipulation techniques. It uses the [Hello Worlds library](https://github.com/kenjinp/hello-worlds) to render the terrain.

The current model is very naïve, expect the output to look crap.

### Example Input

Terrain elevation data input
<br/>
<img src="https://github.com/kenjinp/terrain-synth/blob/main/media/example-model-input.png?raw=true" alt="real earth terrain heightmap example used as input in GAN model" height='256px'/>

### Example Output

(Silly) generated terrain from the [pytorch GAN model](https://github.com/kenjinp/terrain-synth/tree/main/model/gan)
<br/>
<img src="https://github.com/kenjinp/terrain-synth/blob/main/media/example-gan-output.png?raw=true" alt="generated terrain heightmap from the GAN model]" height='256px'/>

## Technologies

- pytorch for ML learning
- onnex runtime (https://onnxruntime.ai/) for model inference
- react / typescript / vite for the app
- [Hello Worlds library](https://github.com/kenjinp/hello-worlds) for terrain rendering
- react-three/fiber / threejs for 3D rendering
- pulumi / aws for deployment

## Datasets

Earth Terrain - [From Kaggle user THOMAS PAPPAS ](https://www.kaggle.com/datasets/tpapp157/earth-terrain-height-and-segmentation-map-images)

<a href='http://www.recurse.com' title='Made with love at the Recurse Center'><img src='https://cloud.githubusercontent.com/assets/2883345/11322972/9e553260-910b-11e5-8de9-a5bf00c352ef.png' height='59px'/></a>
