# Audio dataset and corresponding functions taken from Patter  https://github.com/ryanleary/patter
# Adapted from https://github.com/NVIDIA/NeMo/tree/r0.9/collections/nemo_asr
# MIT License
# Copyright (c) 2020 Xilinx (Giuseppe Franco)
# Copyright (c) 2019 NVIDIA Corporation
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import string

from .cleaners import clean_text


class ManifestBase:
    def __init__(self,
                 manifest_paths,
                 labels,
                 max_duration=None,
                 min_duration=None,
                 sort_by_duration=False,
                 max_utts=0,
                 blank_index=-1,
                 unk_index=-1,
                 normalize=True,
                 logger=None):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sort_by_duration = sort_by_duration
        self.max_utts = max_utts
        self.blank_index = blank_index
        self.unk_index = unk_index
        self.normalize = normalize
        self.labels_map = {label: i for i, label in enumerate(labels)}
        self.logger = None
        # if logger is None:
        #     self.logger = get_logger('')

        data = []
        duration = 0.0
        filtered_duration = 0.0

        for item in self.json_item_gen(manifest_paths):
            if min_duration and item['duration'] < min_duration:
                filtered_duration += item['duration']
                continue
            if max_duration and item['duration'] > max_duration:
                filtered_duration += item['duration']
                continue

            # load and normalize transcript text, i.e. `text`
            text = ""
            if 'text' in item:
                text = item['text']
            elif 'text_filepath' in item:
                text = self.load_transcript(item['text_filepath'])
            else:
                filtered_duration += item['duration']
                continue
            if normalize:
                text = self.normalize_text(text, labels, logger=self.logger)
            if not isinstance(text, str):
                self.logger.warning(
                    "WARNING: Got transcript: {}. It is not a "
                    "string. Dropping data point".format(text)
                )
                filtered_duration += item['duration']
                continue
            # item['text'] = text

            # tokenize transcript text
            item["tokens"] = self.tokenize_transcript(
                    text, self.labels_map, self.unk_index, self.blank_index)

            # support files using audio_filename
            if 'audio_filename' in item and 'audio_filepath' not in item:
                self.logger.warning(
                    "Malformed manifest: The key audio_filepath was not "
                    "found in the manifest. Using audio_filename instead."
                )
                item['audio_filepath'] = item['audio_filename']

            data.append(item)
            duration += item['duration']

            if max_utts > 0 and len(data) >= max_utts:
                self.logger.info(
                    'Stop parsing due to max_utts ({})'.format(max_utts))
                break

        if sort_by_duration:
            data = sorted(data, key=lambda x: x['duration'])
        self._data = data
        self._size = len(data)
        self._duration = duration
        self._filtered_duration = filtered_duration

    @staticmethod
    def normalize_text(text, labels):
        """for the base class remove surrounding whitespace only"""
        return text.strip()

    @staticmethod
    def tokenize_transcript(transcript, labels_map, unk_index, blank_index):
        """tokenize transcript to convert words/characters to indices"""
        # allow for special labels such as "<NOISE>"
        special_labels = set([l for l in labels_map.keys() if len(l) > 1])
        tokens = []
        # split by word to find special tokens
        for i, word in enumerate(transcript.split(" ")):
            if i > 0:
                tokens.append(labels_map.get(" ", unk_index))
            if word in special_labels:
                tokens.append(labels_map.get(word))
                continue
            # split by character to get the rest of the tokens
            for char in word:
                tokens.append(labels_map.get(char, unk_index))
        # if unk_index == blank_index, OOV tokens are removed from transcript
        tokens = [x for x in tokens if x != blank_index]
        return tokens

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._data)

    @staticmethod
    def json_item_gen(manifest_paths):
        for manifest_path in manifest_paths:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    yield json.loads(line)

    @staticmethod
    def load_transcript(transcript_path):
        with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return transcript

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration

    @property
    def data(self):
        return list(self._data)


class ManifestEN(ManifestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def normalize_text(text, labels, logger=None):
        # Punctuation to remove
        punctuation = string.punctuation
        # Define punctuation that will be handled by text cleaner
        punctuation_to_replace = {
            "+": "plus",
            "&": "and",
            "%": "percent"
        }
        for char in punctuation_to_replace:
            punctuation = punctuation.replace(char, "")
        # We might also want to consider:
        # @ -> at
        # -> number, pound, hashtag
        # ~ -> tilde
        # _ -> underscore

        # If a punctuation symbol is inside our vocab, we do not remove
        # from text
        for l in labels:
            punctuation = punctuation.replace(l, "")

        # Turn all other punctuation to whitespace
        table = str.maketrans(punctuation, " " * len(punctuation))

        try:
            text = clean_text(text, table, punctuation_to_replace)
        except BaseException:
            if logger:
                logger.warning("WARNING: Normalizing {} failed".format(text))
            else:
                print("WARNING: Normalizing {} failed".format(text))
            return None

        return text
