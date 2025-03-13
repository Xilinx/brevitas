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

import json
import logging
import os
import pathlib
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import open_clip
from PIL import Image
from scipy import linalg
import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as TF
from tqdm import tqdm

from brevitas_examples.stable_diffusion.mlperf_evaluation.backend import BackendPytorch
from brevitas_examples.stable_diffusion.mlperf_evaluation.backend import Item
from brevitas_examples.stable_diffusion.mlperf_evaluation.backend import RunnerBase
from brevitas_examples.stable_diffusion.mlperf_evaluation.dataset import Coco
from brevitas_examples.stable_diffusion.mlperf_evaluation.dataset import ImagesDataset
from brevitas_examples.stable_diffusion.mlperf_evaluation.inception import InceptionV3

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class CLIPEncoder(nn.Module):
    """
    A class for encoding images and texts using a specified CLIP model and computing the similarity between them.

    Attributes:
    -----------
    clip_version: str
        The version of the CLIP model to be used.
    pretrained: str
        The pre-trained weights to load.
    model: nn.Module
        The CLIP model.
    preprocess: Callable
        The preprocessing transform to apply to the input image.
    device: str
        The device to which the model is moved.
    """

    def __init__(
            self,
            clip_version: str = 'ViT-B/32',
            pretrained: Optional[str] = '',
            cache_dir: Optional[str] = None,
            device: str = 'cpu'):
        """
        Initializes the CLIPEncoder with the specified CLIP model version and pre-trained weights.

        Parameters:
        -----------
        clip_version: str, optional
            The version of the CLIP model to be used. Defaults to 'ViT-B/32'.
        pretrained: str, optional
            The pre-trained weights to load. If not provided, it defaults based on clip_version.
        cache_dir: str, optional
            The directory to cache the model. Defaults to None.
        device: str, optional
            The device to which the model is moved. Defaults to 'cuda'.
        """
        super().__init__()

        self.clip_version = clip_version
        self.pretrained = pretrained if pretrained else self._get_default_pretrained()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.clip_version,
                                                                               pretrained=self.pretrained,
                                                                               cache_dir=cache_dir)

        self.model.eval()
        self.model.to(device)
        self.device = device

    def _get_default_pretrained(self) -> str:
        """Returns the default pretrained weights based on the clip_version."""
        if self.clip_version == 'ViT-H-14':
            return 'laion2b_s32b_b79k'
        elif self.clip_version == 'ViT-g-14':
            return 'laion2b_s12b_b42k'
        else:
            return 'openai'

    @torch.no_grad()
    def get_clip_score(
            self, text: Union[str, List[str]], image: Union[Image.Image,
                                                            torch.Tensor]) -> torch.Tensor:
        """
        Computes the similarity score between the given text(s) and image using the CLIP model.

        Parameters:
        -----------
        text: Union[str, List[str]]
            The text or list of texts to compare with the image.
        image: Image.Image
            The input image.

        Returns:
        --------
        torch.Tensor
            The similarity score between the text(s) and image.
        """
        # Preprocess the image and move it to the specified device
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        # Normalize the image features
        image_features = self.model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # If a single text string is provided, convert it to a list
        if not isinstance(text, (list, tuple)):
            text = [text]

        # Tokenize the text and move it to the specified device
        text = open_clip.tokenize(text).to(self.device)

        # Normalize the text features
        text_features = self.model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the similarity between the image and text features
        similarity = image_features @ text_features.T

        return similarity


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (mu1.shape == mu2.shape), "Training and test mean vectors have different lengths"
    assert (sigma1.shape == sigma2.shape), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_activations(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print((
            "Warning: batch size is bigger than the data size. "
            "Setting batch size to data size"))
        batch_size = len(files)

    dataset = ImagesDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(
        files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(
        path,
        model,
        batch_size,
        dims,
        device,
        num_workers=1,
        subset_size=None,
        shuffle_seed=None,
        ds=None):
    if path.endswith(".npz"):
        with np.load(path) as f:
            m, s = f["mu"][:], f["sigma"][:]
    else:
        path = pathlib.Path(path)
        files = [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]

        files = ds.get_imgs([i for i in range(10)])
        files = [file.permute(1, 2, 0).numpy() for file in files]
        if subset_size is not None:
            random.seed(shuffle_seed)
            files = random.sample(files, subset_size)
        m, s = calculate_activation_statistics(files, model, batch_size, dims, device, num_workers)

    return m, s


def compute_fid(
    results,
    statistics_path,
    device,
    dims=2048,
    num_workers=1,
    batch_size=1,
    subset_size=None,
    shuffle_seed=None,
    ds=None,
):
    imgs = [Image.fromarray(e).convert("RGB") for e in results]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = num_workers
    # assert statistics_path.endswith(".npz")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(
        statistics_path,
        model,
        batch_size,
        dims,
        device,
        num_workers,
        subset_size,
        shuffle_seed,
        ds=ds
    )

    m2, s2 = calculate_activation_statistics(imgs, model, batch_size, dims, device, num_workers)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


class PostProcessCoco:

    def __init__(
        self,
        device="cpu",
        dtype="uint8",
        statistics_path=os.path.join(os.path.dirname(__file__), "tools", "val2014.npz")):
        self.results = []
        self.good = 0
        self.total = 0
        self.content_ids = []
        self.clip_scores = []
        self.fid_scores = []
        self.device = device if torch.cuda.is_available() else "cpu"
        if dtype == "uint8":
            self.dtype = torch.uint8
            self.numpy_dtype = np.uint8
        else:
            raise ValueError(f"dtype must be one of: uint8")
        self.statistics_path = statistics_path

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, ids, expected=None, result_dict=None):
        self.content_ids.extend(ids)
        return [(t.cpu().permute(1, 2, 0).float().numpy() * 255).round().astype(self.numpy_dtype)
                for t in results]

    def save_images(self, ids, ds):
        info = []
        idx = {}
        for i, id in enumerate(self.content_ids):
            if id in ids:
                idx[id] = i
        if not os.path.exists("images/"):
            os.makedirs("images/", exist_ok=True)
        for id in ids:
            caption = ds.get_caption(id)
            generated = Image.fromarray(self.results[idx[id]])
            image_path_tmp = f"images/{self.content_ids[idx[id]]}.png"
            generated.save(image_path_tmp)
            info.append((self.content_ids[idx[id]], caption))
        with open("images/captions.txt", "w+") as f:
            for id, caption in info:
                f.write(f"{id}  {caption}\n")

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None, output_dir=None):
        clip = CLIPEncoder(device=self.device)
        dataset_size = len(self.results)
        log.info("Accumulating results")
        for i in range(0, dataset_size):
            caption = ds.get_caption(self.content_ids[i])
            generated = Image.fromarray(self.results[i])
            self.clip_scores.append(100 * clip.get_clip_score(caption, generated).item())

        fid_score = compute_fid(self.results, self.statistics_path, self.device, ds=ds)
        result_dict["FID_SCORE"] = fid_score
        result_dict["CLIP_SCORE"] = np.mean(self.clip_scores)
        return result_dict


def compute_mlperf_fid(
        path_to_sdxl,
        path_to_coco,
        model_to_replace=None,
        samples_to_evaluate=500,
        output_dir=None,
        device='cpu',
        vae_force_upcast=True):

    assert os.path.isfile(path_to_coco + '/tools/val2014.npz'), "Val2014.npz file required. Check the MLPerf directory for instructions"

    post_proc = PostProcessCoco(statistics_path=path_to_coco + '/tools/val2014.npz', device=device)

    dtype = next(iter(model_to_replace.unet.parameters())).dtype
    res_dict = {}
    model = BackendPytorch(
        path_to_sdxl, 'xl', steps=20, batch_size=1, device=device, precision=dtype)
    model.load()

    if model_to_replace is not None:
        model.pipe.unet = model_to_replace.unet
        if not vae_force_upcast:
            model.pipe.vae = model_to_replace.vae

    model.pipe.vae.config.force_upcast = vae_force_upcast
    ds = Coco(
        data_path=path_to_coco,
        name="coco-1024",
        pre_process=torch.nn.Identity,
        count=None,
        threads=1,
        pipe_tokenizer=model.pipe.tokenizer,
        pipe_tokenizer_2=model.pipe.tokenizer_2,
        latent_dtype=dtype,
        latent_device='cuda',
        latent_framework='torch',
        **{"image_size": [3, 1024, 1024]},
    )
    model.pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        runner = RunnerBase(model, ds, 1, post_proc=post_proc, max_batchsize=1)
        runner.start_run(res_dict, True)
        idx = list(range(0, samples_to_evaluate))
        ds.load_query_samples(idx)
        data, label = ds.get_samples(idx)
        runner.run_one_item(Item(idx, idx, data, label))
        post_proc.finalize(res_dict, ds=ds)
        log.info(res_dict)
    if output_dir is not None:
        # Dump args to json
        with open(os.path.join(output_dir, 'results_mlperf.json'), 'w') as fp:
            json.dump(res_dict, fp)
