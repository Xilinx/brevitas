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
import array
import logging
import time
from typing import Optional

from diffusers import EulerDiscreteScheduler
from diffusers import StableDiffusionXLPipeline
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, inputs, img=None):
        self.query_id = query_id
        self.content_id = content_id
        self.img = img
        self.inputs = inputs
        self.start = time.time()


class RunnerBase:

    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        self.take_accuracy = False
        self.ds = ds
        self.model = model
        self.post_process = post_proc
        self.threads = threads
        self.take_accuracy = False
        self.max_batchsize = max_batchsize
        self.result_timing = []

    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.result_timing = []
        self.take_accuracy = take_accuracy
        self.post_process.start()

    def run_one_item(self, qitem: Item):
        # run the prediction
        processed_results = []
        try:
            results = self.model.predict(qitem.inputs)
            processed_results = self.post_process(
                results, qitem.content_id, qitem.inputs, self.result_dict)
            if self.take_accuracy:
                self.post_process.add_results(processed_results)
            self.result_timing.append(time.time() - qitem.start)
        except Exception as ex:  # pylint: disable=broad-except
            src = [self.ds.get_item_loc(i) for i in qitem.content_id]
            log.error("thread: failed on contentid=%s, %s", src, ex)
            # since post_process will not run, fake empty responses
            processed_results = [[]] * len(qitem.query_id)
        finally:
            response_array_refs = []
            response = []
            for idx, query_id in enumerate(qitem.query_id):
                response_array = array.array(
                    "B", np.array(processed_results[idx], np.uint8).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append((query_id, bi[0], bi[1]))
            # lg.QuerySamplesComplete(response)

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            data, label = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, data, label))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                data, label = self.ds.get_samples(idx[i:i + bs])
                self.run_one_item(Item(query_id[i:i + bs], idx[i:i + bs], data, label))

    def finish(self):
        pass


class BackendPytorch:

    def __init__(
        self,
        model_path=None,
        model_id="xl",
        guidance=8,
        steps=20,
        batch_size=1,
        device="cuda",
        precision=torch.float32,
        negative_prompt="normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
    ):
        self.inputs = []
        self.outputs = []

        self.model_path = model_path
        if model_id == "xl":
            self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            raise ValueError(f"{model_id} is not a valid model id")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = precision

        if torch.cuda.is_available():
            self.local_rank = 0
            self.world_size = 1

        self.guidance = guidance
        self.steps = steps
        self.negative_prompt = negative_prompt
        self.max_length_neg_prompt = 77
        self.batch_size = batch_size

    def version(self):
        return torch.__version__

    # def name(self):
    #     return "pytorch-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        if self.model_path is None:
            log.warning(
                "Model path not provided, running with default hugging face weights\n"
                "This may not be valid for official submissions")
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                self.model_id, subfolder="scheduler")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                scheduler=self.scheduler,
                safety_checker=None,
                add_watermarker=False,
                variant="fp16" if (self.dtype == torch.float16) else None,
                torch_dtype=self.dtype,
            )
            # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        else:
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                self.model_id, subfolder="scheduler")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_path,
                scheduler=self.scheduler,
                safety_checker=None,
                add_watermarker=False,
                variant="fp16" if (self.dtype == torch.float16) else None,
                torch_dtype=self.dtype,
            )
            # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        self.pipe.to(self.device)
        #self.pipe.set_progress_bar_config(disable=True)

        self.negative_prompt_tokens = self.pipe.tokenizer(
            self.convert_prompt(self.negative_prompt, self.pipe.tokenizer),
            padding="max_length",
            max_length=self.max_length_neg_prompt,
            truncation=True,
            return_tensors="pt",
        )
        self.negative_prompt_tokens_2 = self.pipe.tokenizer_2(
            self.convert_prompt(self.negative_prompt, self.pipe.tokenizer_2),
            padding="max_length",
            max_length=self.max_length_neg_prompt,
            truncation=True,
            return_tensors="pt",
        )
        return self

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

    def encode_tokens(
        self,
        pipe: StableDiffusionXLPipeline,
        text_input: torch.Tensor,
        text_input_2: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[torch.Tensor] = None,
        negative_prompt_2: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the input tokens into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or pipe._execution_device
        batch_size = text_input.input_ids.shape[0]

        # Define tokenizers and text encoders
        tokenizers = ([pipe.tokenizer, pipe.tokenizer_2] if pipe.tokenizer is not None else [
            pipe.tokenizer_2])
        text_encoders = ([pipe.text_encoder, pipe.text_encoder_2]
                         if pipe.text_encoder is not None else [pipe.text_encoder_2])

        if prompt_embeds is None:
            text_input_2 = text_input_2 or text_input

            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            text_inputs_list = [text_input, text_input_2]
            for text_inputs, tokenizer, text_encoder in zip(
                text_inputs_list, tokenizers, text_encoders
            ):
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = (
            negative_prompt is None and pipe.config.force_zeros_for_empty_prompt)
        if (do_classifier_free_guidance and negative_prompt_embeds is None and
                zero_out_negative_prompt):
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt_inputs = (
                negative_prompt.input_ids.repeat(batch_size, 1) if
                (len(negative_prompt.input_ids.shape) == 1) else negative_prompt.input_ids)
            negative_prompt_2_inputs = (
                negative_prompt_2.input_ids.repeat(batch_size, 1) if
                (len(negative_prompt_2.input_ids.shape) == 1) else negative_prompt_2.input_ids)

            uncond_inputs = [negative_prompt_inputs, negative_prompt_2_inputs]

            negative_prompt_embeds_list = []
            for uncond_input, tokenizer, text_encoder in zip(
                uncond_inputs, tokenizers, text_encoders
            ):
                negative_prompt_embeds = text_encoder(
                    uncond_input.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if pipe.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=pipe.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if pipe.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=pipe.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=pipe.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1)
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_inputs(self, inputs, i):
        if self.batch_size == 1:
            return self.encode_tokens(
                self.pipe,
                inputs[i]["input_tokens"],
                inputs[i]["input_tokens_2"],
                negative_prompt=self.negative_prompt_tokens,
                negative_prompt_2=self.negative_prompt_tokens_2,
            )
        else:
            prompt_embeds = []
            negative_prompt_embeds = []
            pooled_prompt_embeds = []
            negative_pooled_prompt_embeds = []
            for prompt in inputs[i:min(i + self.batch_size, len(inputs))]:
                assert isinstance(prompt, dict)
                text_input = prompt["input_tokens"]
                text_input_2 = prompt["input_tokens_2"]
                (
                    p_e,
                    n_p_e,
                    p_p_e,
                    n_p_p_e,
                ) = self.encode_tokens(
                    self.pipe,
                    text_input,
                    text_input_2,
                    negative_prompt=self.negative_prompt_tokens,
                    negative_prompt_2=self.negative_prompt_tokens_2,
                )
                prompt_embeds.append(p_e)
                negative_prompt_embeds.append(n_p_e)
                pooled_prompt_embeds.append(p_p_e)
                negative_pooled_prompt_embeds.append(n_p_p_e)

            prompt_embeds = torch.cat(prompt_embeds)
            negative_prompt_embeds = torch.cat(negative_prompt_embeds)
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds)
            negative_pooled_prompt_embeds = torch.cat(negative_pooled_prompt_embeds)
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def predict(self, inputs):
        images = [0] * len(inputs)
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), self.batch_size)):
                max_index = min(i + self.batch_size, len(inputs))
                latents_input = [inputs[idx]["latents"] for idx in range(i, max_index)]
                latents_input = torch.cat(latents_input).to(self.device)
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.prepare_inputs(inputs, i)
                generated = self.pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    guidance_scale=self.guidance,
                    num_inference_steps=self.steps,
                    output_type="pt",
                    latents=latents_input,
                ).images
                images[i:i + max_index] = generated.cpu()
        return images
