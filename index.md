# Combining audio control and style transfer using latent diffusion 


**Abstract**
Deep generative models are now able to synthesize high-quality audio signals, shifting the critical aspect in their development from audio quality to control capabilities. Although text-to-music generation is getting largely adopted by the general public, explicit control and example-based style transfer are more adequate modalities to capture the intents of artists and musicians. 
In this paper, we aim to unify explicit control and style transfer within a single model by separating local and global information to capture musical structure and timbre respectively. To do so, we leverage the capabilities of diffusion autoencoders to extract semantic features, in order to build two representation spaces. We enforce disentanglement between those spaces using an adversarial criterion and a two-stage training strategy. Our resulting model can generate audio matching a timbre target, while specifying structure either with explicit controls or through another audio example. We evaluate our model on one-shot timbre transfer and MIDI-to-audio tasks on instrumental recordings and show that we outperform existing baselines in terms of audio quality and target fidelity. Furthermore, we show that our method can generate cover versions of complete musical pieces by transferring rhythmic and melodic content to the style of a target audio in a different genre. 

<img src="images/method.png">


# MIDI-to-audio
## Reconstruction


| | <div style="width:290px"></div>  | original | midi synthesis | SpecDiff | Ours with encoder | Ours |
| :-:| :-:  | :-: | :-: | :-: | :-: | :-: |
| Piano  | <img src="images/midi-to-audio/1.png" alt="drawing" controls style="width:  300px" >  |<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | 
| Violin  |  <img src="images/midi-to-audio/1.png" alt="drawing" controls style="width:  300px" >  | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
| Guitar   | <img src="images/midi-to-audio/1.png" alt="drawing" controls style="width:  300px" >  | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
| Guitar   | <img src="images/midi-to-audio/1.png" alt="drawing" controls style="width:  300px" >  | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |


## Changing the target instrument


| | <div style="width:290px"></div>  | original | SpecDiff | Ours with encoder | Ours |
| :-:| :-:  | :-: | :-: | :-: | :-: |
| Piano to guitar  | <img src="images/midi-to-audio/1.png" alt="drawing" controls style="width:  300px" >  |<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
| Piano to guitar  | <img src="images/midi-to-audio/1.png" alt="drawing" controls style="width:  300px" >  |<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |




# Timbre Transfer
## Synthetic Data


| | Source | Target | <div style="width:60px"></div>| SS-VAE | Music Style Transfer | Ours no adv. | Ours |
| :-:| :-: | :-:  |:-:  | :-: | :-: | :-: | :-: |
  | Piano 1|<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
  ||| <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
||| <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
| Violin 1|<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
||| <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
||| <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
| Guitar|<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
||| <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |
||| <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |





## Real Data


| Source | Target | <div style="width:60px"></div>| SS-VAE | Music Style Transfer | Ours no adv. | Ours |
| :-: | :-:  |:-:  | :-: | :-: | :-: | :-: |
  |<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> |





  # Music style transfer


  | Source | Target | <div style="width:60px"></div>| MusicGen | Ours no adv. | Ours |
| :-: | :-:  |:-:  | :-: | :-: | :-: | 
|<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | 
|<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | 
|<audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/x.mp3" controls style="width:  120px"></audio> | <audio src="eval_timbre_2/y.mp3" controls style="width:  120px"></audio> | 

