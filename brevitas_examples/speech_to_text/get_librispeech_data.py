# Adapted from https://github.com/NVIDIA/NeMo/tree/r0.9
# Copyright (C) 2020 Xilinx (Giuseppe Franco)
# Copyright (C) 2019 NVIDIA CORPORATION.
#
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import fnmatch
import json
import os
import subprocess
import tarfile
import urllib.request

from sox import Transformer
from tqdm import tqdm

parser = argparse.ArgumentParser(description='LibriSpeech Data download')
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--data_sets", default="dev_clean", type=str)
args = parser.parse_args()

URLS = {
    'TRAIN_CLEAN_100': ("http://www.openslr.org/resources/12/train-clean-100.tar.gz"),
    'TRAIN_CLEAN_360': ("http://www.openslr.org/resources/12/train-clean-360.tar.gz"),
    'TRAIN_OTHER_500': ("http://www.openslr.org/resources/12/train-other-500.tar.gz"),
    'DEV_CLEAN': "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    'DEV_OTHER': "http://www.openslr.org/resources/12/dev-other.tar.gz",
    'TEST_CLEAN': "http://www.openslr.org/resources/12/test-clean.tar.gz",
    'TEST_OTHER': "http://www.openslr.org/resources/12/test-other.tar.gz",
}


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource

    Returns:

    """
    source = URLS[source]
    if not os.path.exists(destination):
        print("{0} does not exist. Downloading ...".format(destination))
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        print("Downloaded {0}.".format(destination))
    else:
        print("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        print('Not extracting. Maybe already there?')


def __process_data(data_folder: str, dst_folder: str, manifest_file: str):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest

    Returns:

    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = []
    entries = []

    for root, dirnames, filenames in os.walk(data_folder):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            files.append((os.path.join(root, filename), root))

    for transcripts_file, root in tqdm(files):
        with open(transcripts_file, encoding="utf-8") as fin:
            for line in fin:
                id, text = line[: line.index(" ")], line[line.index(" ") + 1 :]
                transcript_text = text.lower().strip()

                # Convert FLAC file to WAV
                flac_file = os.path.join(root, id + ".flac")
                wav_file = os.path.join(dst_folder, id + ".wav")
                if not os.path.exists(wav_file):
                    Transformer().build(flac_file, wav_file)
                # check duration
                duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

                entry = dict()
                entry['audio_filepath'] = os.path.abspath(wav_file)
                entry['duration'] = float(duration)
                entry['text'] = transcript_text
                entries.append(entry)

    with open(manifest_file, 'w') as fout:
        for m in entries:
            fout.write(json.dumps(m) + '\n')


def main():
    data_root = args.data_root
    data_sets = args.data_sets

    if data_sets == "ALL":
        data_sets = "dev_clean,dev_other,train_clean_100,train_clean_360,train_other_500,test_clean,test_other"

    for data_set in data_sets.split(','):
        print("\n\nWorking on: {0}".format(data_set))
        filepath = os.path.join(data_root, data_set + ".tar.gz")
        print("Getting {0}".format(data_set))
        __maybe_download_file(filepath, data_set.upper())
        print("Extracting {0}".format(data_set))
        __extract_file(filepath, data_root)
        print("Processing {0}".format(data_set))
        __process_data(
            os.path.join(os.path.join(data_root, "LibriSpeech"), data_set.replace("_", "-"),),
            os.path.join(os.path.join(data_root, "LibriSpeech"), data_set.replace("_", "-"),) + "-processed",
            os.path.join(data_root, data_set + ".json"),
        )
    print('Done!')


if __name__ == "__main__":
    main()