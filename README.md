# Combining audio control and style transfer using latent diffusion

Official repository for the paper "Combining audio control and style transfer using latent diffusion", accepted at ISMIR 2024.

Training the model requires three steps : processing the dataset, training an autoencoder, then training the diffusion model.


### Dataset preparation

```bash
python dataset/split_to_lmdb.py --input_path /path/to/audio_dataset --output_path /path/to/audio_dataset/out_lmdb
```

Or to use slakh with midi processing (after downloading Slakh2100  [here](http://www.slakh.com/)) :

```bash
python dataset/split_to_lmdb_midi.py --input_path /path/to/slakh --output_path /path/to/slakh/out_lmdb_midi --slakh True
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

### Model training
The model training is configured with gin config files.
To train the audio to audio model :
```bash
 python train_diffusion.py --db_path /data/nils/datasets/slakh/lmdb_midi/ --config midi --dataset_type midi --gpu #
```
To train the midi-to-audio model : 
```bash
 python train_diffusion.py --db_path /path/to/lmdb --config main --dataset_type waveform --gpu #
```

### Inference and evaluation
TBA
