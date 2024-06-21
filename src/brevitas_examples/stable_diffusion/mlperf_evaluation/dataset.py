"""
Code is adapted from the MLPerf text-to-image pipeline: https://github.com/mlcommons/inference/tree/master/text_to_image
Available under the following LICENSE:

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
"""

import logging
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class Dataset:

    def __init__(self):
        self.arrival = None
        self.image_list = []
        self.caption_list = []
        self.items_inmemory = {}
        self.last_loaded = -1

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return len(self.image_list)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        self.items_inmemory = {}
        for sample in sample_list:
            self.items_inmemory[sample] = self.get_item(sample)
        self.last_loaded = time.time()

    def unload_query_samples(self, sample_list):
        if sample_list:
            for sample in sample_list:
                if sample in self.items_inmemory:
                    del self.items_inmemory[sample]
        else:
            self.items_inmemory = {}

    def get_samples(self, id_list):
        data = [{
            "input_tokens": self.items_inmemory[id]["input_tokens"],
            "input_tokens_2": self.items_inmemory[id]["input_tokens_2"],
            "latents": self.items_inmemory[id]["latents"],} for id in id_list]
        images = [self.items_inmemory[id]["file_name"] for id in id_list]
        return data, images

    def get_item(self, id):
        raise NotImplementedError("Dataset:get_item")


class ImagesDataset(torch.utils.data.Dataset):

    def __init__(self, imgs, transforms=None):
        self.imgs = imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class Coco(Dataset):

    def __init__(
        self,
        data_path,
        name=None,
        image_size=None,
        pre_process=None,
        pipe_tokenizer=None,
        pipe_tokenizer_2=None,
        latent_dtype=torch.float32,
        latent_device="cuda",
        latent_framework="torch",
        **kwargs,
    ):
        super().__init__()
        self.captions_df = pd.read_csv(f"{data_path}/captions/captions.tsv", sep="\t")
        self.image_size = image_size
        self.preprocessed_dir = os.path.abspath(f"{data_path}/preprocessed/")
        self.img_dir = os.path.abspath(f"{data_path}/validation/data/")
        self.name = name

        # Preprocess prompts
        self.captions_df["input_tokens"] = self.captions_df["caption"].apply(
            lambda x: self.preprocess(x, pipe_tokenizer))
        self.captions_df["input_tokens_2"] = self.captions_df["caption"].apply(
            lambda x: self.preprocess(x, pipe_tokenizer_2))
        self.latent_dtype = latent_dtype
        self.latent_device = latent_device if torch.cuda.is_available() else "cpu"
        if latent_framework == "torch":
            self.latents = (
                torch.load(f"{data_path}/latents/latents.pt").to(latent_dtype).to(latent_device))
        elif latent_framework == "numpy":
            self.latents = (
                torch.Tensor(
                    np.load(f"{data_path}/latents/latents.npy")).to(latent_dtype).to(latent_device))

    def preprocess(self, prompt, tokenizer):
        converted_prompt = self.convert_prompt(prompt, tokenizer)
        return tokenizer(
            converted_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def image_to_tensor(self, img):
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        tensor = torch.Tensor(img.transpose([2, 0, 1])).to(torch.uint8)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    def preprocess_images(self, file_name):
        img = Image.open(self.img_dir + "/" + file_name)
        tensor = self.image_to_tensor(img)
        target_name = file_name.split(".")[0]
        target_path = self.preprocessed_dir + "/" + target_name + ".pt"
        if not os.path.exists(target_path):
            torch.save(tensor, target_path)
        return target_path

    def convert_prompt(self, prompt, tokenizer):
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def get_item(self, id):
        return dict(self.captions_df.loc[id], latents=self.latents)

    def get_item_count(self):
        return len(self.captions_df)

    def get_img(self, id):
        img = Image.open(self.img_dir + "/" + self.captions_df.loc[id]["file_name"])
        return self.image_to_tensor(img)

    def get_imgs(self, id_list):
        image_list = []
        for id in id_list:
            image_list.append(self.get_img(id))
        return image_list

    def get_caption(self, i):
        return self.get_item(i)["caption"]

    def get_captions(self, id_list):
        return [self.get_caption(id) for id in id_list]

    def get_item_loc(self, id):
        return self.img_dir + "/" + self.captions_df.loc[id]["file_name"]
