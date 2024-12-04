# Combining audio control and style transfer using latent diffusion

Official repository for  _Combining audio control and style transfer using latent diffusion_ by Nils Demerlé, Philippe Esling, Guillaume Doras and David Genova accepted at ISMIR 2024 ([paper link](https://arxiv.org/pdf/2408.00196)).

This diffusion-based generative model creates new audio by blending two inputs: one audio sample that sets the style or timbre, and another input (either audio or MIDI) that defines the structure over time. In this repository, you will find instructions to train your own model as well as model checkpoints trained on the two datasets presented in the paper. 

We are currently working on a real-time implementation of this model called AFTER. You can already experiment with a real-time version of the model in MaxMSP on the official AFTER [repository](https://github.com/acids-ircam/AFTER). 



## Model training
Training the model requires three steps : processing the dataset, training an autoencoder, then training the diffusion model.

### Dataset preparation

```bash
python dataset/split_to_lmdb.py --input_path /path/to/audio_dataset --output_path /path/to/audio_dataset/out_lmdb
```

Or to use slakh with midi processing (after downloading Slakh2100  [here](http://www.slakh.com/)) :

```bash
python dataset/split_to_lmdb.py --input_path /path/to/slakh --output_path /path/to/slakh/out_lmdb_midi --slakh True
```

### Autoencoder training

```bash
python train_autoencoder.py --name my_autoencoder --db_path /path/to/lmdb --gpu #
```
Once the autoencoder is trained, it must be exported to a torchscript .pt file : 

```bash
 python export_autoencoder.py --name my_autoencoder --step ##
```

It is possible to skip this whole phase and use a pretrained autoencoder such as Encodec, wrapped in a nn.module with encode and decode methods. 

### Diffusion model training
The model training is configured with gin config files.

To train the audio to audio model :
```bash
 python train_diffusion.py --name my_audio_model --db_path /path/to/lmdb --config main --dataset_type waveform --gpu #
```

To train the midi-to-audio model : 
```bash
 python train_diffusion.py --name my_midi_audio_model  --db_path /path/to/lmdb_midi --config midi --dataset_type midi --gpu #
```

## Inference and pretrained models

Three pretrained models are currently available : 
1. Audio to audio transfer model trained on [Slakh](http://www.slakh.com/)
2. Audio to audio transfer model trained on multiple datasets (Maestro, URMP, Filobass, GuitarSet...)
3. MIDI-to-audio model trained on [Slakh](http://www.slakh.com/)

You can download the autoencoder and diffusion model checkpoints [here](https://nubo.ircam.fr/index.php/s/8xaXbQtcY4n3Mg9/download). Make sure you copy the pretrained models in `./pretrained`. The notebooks in `./notebooks` demonstrate how to load a model and generate audio from midi and audio files.
