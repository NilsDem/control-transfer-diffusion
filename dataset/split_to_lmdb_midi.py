# adapted from https://github.com/acids-ircam/RAVE/blob/master/scripts/preprocess.py

import functools
import multiprocessing
import os
import pathlib
import subprocess
from datetime import timedelta
from functools import partial
from itertools import repeat
from typing import Callable, Iterable, Sequence, Tuple
import yaml
from tqdm import tqdm
import lmdb
import numpy as np
import torch
from absl import app, flags

from audio_example import AudioExample
import numpy as np
import sys
import pretty_midi


torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path',
                          None,
                          help='Path to a directory containing audio files - use slakh main directory to use slakh',
                          required=True)
flags.DEFINE_string('output_path',
                    ".",
                    help='Output directory for the dataset',
                    required=False)

flags.DEFINE_bool('slakh',
                  False,
                  help='use slakh data processing',
                  required=False)

flags.DEFINE_integer('num_signal',
                     262144,
                     help='Number of audio samples to use during training')

flags.DEFINE_integer('num_cores',
                     8,
                     help='Number of cores for multiprocessing')

flags.DEFINE_integer('sampling_rate',
                     24000,
                     help='Sampling rate to use during training')
flags.DEFINE_integer('max_db_size',
                     180,
                     help='Maximum size (in GB) of the dataset')

flags.DEFINE_integer('max_files',
                     None,
                     help='Take a random subset of the files')


flags.DEFINE_multi_string(
    'ext',
    default=['wav', 'opus', 'mp3', 'aac', 'flac'],
    help='Extension to search for in the input directory')


flags.DEFINE_bool('dyndb',
                  default=False,
                  help="Allow the database to grow dynamically")




def float_array_to_int16_bytes(x):
    return np.floor(x * (2**15 - 1)).astype(np.int16).tobytes()


def load_audio_chunk(audio_file: tuple, n_signal: int,
                     sr: int) -> Iterable[np.ndarray]:

    path, metadata = audio_file

    process = subprocess.Popen(
        [
            'ffmpeg', '-hide_banner', '-loglevel', 'panic', '-i', path, '-ac',
            '1', '-ar',
            str(sr), '-f', 's16le', '-'
        ],
        stdout=subprocess.PIPE,
    )

    chunk = process.stdout.read(2 * n_signal)
    i = 0
    while len(chunk) == 2 * n_signal:
        metadata["chunk_number"] = i
        i += 1
        yield chunk, metadata
        chunk = process.stdout.read(2 * n_signal)

    process.stdout.close()
    

def get_midi(metadata):
    # MIDI
    path = metadata["path"]
    split = path.split("/")
    split[-2] = "MIDI"

    midi_path = "/".join(split)[:-5] + ".mid"
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    length = FLAGS.num_signal
    tstart = metadata["chunk_number"] * length / 24000
    tend = (metadata["chunk_number"] + 1) * length / 24000


    out_notes = []
    for note in midi_data.instruments[0].notes:
        if note.end > tstart and note.start < tend:
            note.start = max(0, note.start - tstart)
            note.end = note.end - tstart
            out_notes.append(note)


    midi_data.instruments[0].notes = out_notes

    #end = out_notes[-1].end
    end = length / 24000
    midi_data.adjust_times([0, end], [0, end])

    return midi_data


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm
    
def process_audio_array(audio: Tuple[int, Tuple[bytes,
                                                dict]], env: lmdb.Environment,
                        n_signal: int) -> int:
    
    audio_id, data = audio
    audio_samples, metadata = data
    
    ae = AudioExample()

    # Get the midi
    midi = get_midi(metadata)
    #print(midi.instruments[0].notes)
    pr = midi.get_piano_roll(times=np.linspace(0, FLAGS.num_signal / 24000, FLAGS.num_signal //
                                               512))
    # Check if no note is played
    if len(midi.instruments[0].notes) == 0:
        return audio_id, False
        
    ae.put_array("pr", pr, dtype=np.int16)
    ae.put_buffer("waveform", audio_samples, [n_signal])
    ae.put_metadata(metadata)
        
        
    key = f'{audio_id:08d}'
    with env.begin(write=True) as txn:
        txn.put(
            key.encode(),
            bytes(ae),
        )
    return audio_id, True


def flatmap(pool: multiprocessing.Pool,
            func: Callable,
            iterable: Iterable,
            chunksize=None):
    queue = multiprocessing.Manager().Queue(maxsize=os.cpu_count())
    pool.map_async(
        functools.partial(flat_mappper, func),
        zip(iterable, repeat(queue)),
        chunksize,
        lambda _: queue.put(None),
        lambda *e: print(e),
    )

    item = queue.get()
    while item is not None:
        yield item
        item = queue.get()


def flat_mappper(func, arg):
    data, queue = arg
    for item in func(data):
        queue.put(item)


def search_for_audios(path_list: Sequence[str], extensions: Sequence[str]):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f'*.{ext}'))
    audios = flatten(audios)
    return audios


def main(dummy):
    FLAGS(sys.argv)

    chunk_load = partial(load_audio_chunk,
                         n_signal=FLAGS.num_signal,
                         sr=FLAGS.sampling_rate)

   # create database
    env = lmdb.open(FLAGS.output_path,
                   map_size=FLAGS.max_db_size * 1024**3,
                   map_async=True,
                   writemap=True,
                   readahead=False)

    print("number of cores: ", FLAGS.num_cores)
    pool = multiprocessing.Pool(FLAGS.num_cores)


    if FLAGS.slakh == True:
        
        print("processing slakh files")
        
        folder = FLAGS.input_path
        print(folder)

        tracks = [
            os.path.join(folder, subfolder) for subfolder in os.listdir(folder)
        ]

        ban_list = [
            "Chromatic Percussion", "Drums", "Percussive", "Sound Effects",
            "Sound effects"
        ]  

        instr = []
        stem_list = []
        metadata = []

        total_stems = 0
        for trackfolder in tqdm(tracks):
            meta = trackfolder + "/metadata.yaml"
            try:
                with open(meta, 'r') as file:
                    d = yaml.safe_load(file)
            except:
                continue

            for k, stem in d["stems"].items():
                if stem["inst_class"] not in ban_list:
                    stem_list.append(trackfolder + "/stems/" + k + ".flac")
                    instr.append(stem["inst_class"])
                    metadata.append(stem)
                total_stems += 1

        print(set(instr), "instruments remaining")
        print(total_stems, "stems in total")
        print(len(stem_list), "stems retained")

        audios = stem_list

        metadata = [{
            "path": audio,
            "instrument": inst
        } for audio, inst in zip(audios, instr)]

    else:
        # search for audio files
        audios = search_for_audios(FLAGS.input_path, FLAGS.ext)
        audios = map(str, audios)
        audios = map(os.path.abspath, audios)
        audios = [*audios]
        metadata = [{"path": audio} for audio in audios]

    

    print(len(audios), " audio files found")
    
    if FLAGS.max_files is not None and len(audios) > FLAGS.max_files:
        indexes = np.random.choice(list(range(len(audios))), FLAGS.max_files, replace=False)
        audios, metadata = np.array(audios)[indexes], np.array(metadata)[indexes]

    audios = list(zip(audios, metadata))
    
    print(len(audios), " audio files retained")
    

    # load chunks
    chunks = flatmap(pool, chunk_load, audios)
    chunks = enumerate(chunks)
    
    print("reading chunks")
    
    processed_samples = map(
        partial(process_audio_array,
                env=env,
                n_signal=FLAGS.num_signal), chunks)
    
    pbar = tqdm(processed_samples)
    print("processing samples")
    
    
    i = 0
    for audio_id in pbar:
        if audio_id[1] == True:
            i += 1
        n_seconds = FLAGS.num_signal / FLAGS.sampling_rate * i

        pbar.set_description(
            f'iter: {audio_id[0]} - idataset length: {timedelta(seconds=n_seconds)}'
        )
    pool.close()
    env.close()


if __name__ == '__main__':
    app.run(main)
