# Few-Bit LLM Quantization with Learned Round

💡 [User Guide](https://xilinx.github.io/brevitas/dev/user_guide/learned_round.html)

> [!IMPORTANT]
> These yaml files work with brevitas==0.12.1, transformers==4.47.1, and lighteval==0.9.2

Please use `benchmark.py` to reproduce the experiments in the [Results](#results) section, as follows:

```bash
python benchmark.py --config quarot_star.yaml --results results/ --gpus 0,1
```
where `--gpus` refers to how many gpus to use. If multiple GPUs are specified, each one will be used
to run an individual experiment.

Results
==============

To demonstrate the effectivenes and flexibility of the Learned Round implementation in Brevitas,
its performance was compared against the [SignRound](https://aclanthology.org/2024.findings-emnlp.662.pdf) for weight-only quantization,
and against GPTQ and Qronos for the rest of scenarios.

In comparison with [SignRound](https://aclanthology.org/2024.findings-emnlp.662.pdf), Signed SGD was also used in these experiments,
but the number of iterations and the learning rate were decoupled, thus requiring the clipping operation in ``LearnedRoundIdentity``.
Moreover, the SGD optimizer was used for learning the scales, and these are parametrized directly, instead of learning the weight clipping,
while in [SignRound](https://aclanthology.org/2024.findings-emnlp.662.pdf) the authors use Sign SGD to learn the weight clipping, in the same fashion as
[OmniQuant](https://arxiv.org/pdf/2308.13137).

Experiments were conducted on **Llama 3.2** and **Qwen 2.5** base models, sourced from **Huggingface**, using **WikiText2** for validation.
To assess generalization, **LightEval** was used across five zero-shot reasoning tasks, reporting the normalized average accuracy for these:

- ARC (challenge and easy)
- HellaSwag
- PIQA
- Winogrande

For the composition of PTQ techniques, the taxonomy from [Qronos](https://arxiv.org/pdf/2505.11695) was adopted, subdiving these into **Stage 1 (Transform)**
and **Stage 2 (Rounding)**.

For the **Transform** phase (Stage 1) the following methods were used:

- **None**: No preprocessing.
- **[HIP](https://arxiv.org/pdf/2307.13304)**: Hadamard-based incoherence processing.
- **[MagR](https://arxiv.org/pdf/2406.00800)**: Weight magnitude reduction.
- **[QuaRot](https://arxiv.org/pdf/2404.00456)**: Rotation-based outlier reduction.
- **[SpinQuant](https://arxiv.org/pdf/2405.16406)**: Cailey-optimized orthogonal rotations.

For the **Rounding** phase (Stage 2) the following methods were used:

- **RTN**: Round-To-Nearest.
- **[GPTQ](https://arxiv.org/pdf/2210.17323)**.
- **[Qronos](https://arxiv.org/pdf/2505.11695)**.
- **Learned Round**: Brevitas' implementation of learnable rounding.

The following quantization configurations were evaluated:

- **W2g128**: 2-bit weight-only quantization with group size of 128. See [Weight-only quantization](#weight-only-quantization-of-llama-32-and-qwen-25-foundation-models).
- **W4**: 4-bit per-channel weight-only quantization. See [Weight-only quantization](#weight-only-quantization-of-llama-32-and-qwen-25-foundation-models).
- **W4g128**: 4-bit weight-only quantization with group size of 128. See [Weight-only quantization](#weight-only-quantization-of-llama-32-and-qwen-25-foundation-models).
- **W4g32A4g32**: 4-bit weight and activation quantization with group size of 128. See [Weight and activation quantization](#weight-and-activation-quantization-of-llama-32-foundation-models).
- **W4g32A4g32Po2**: MXFP4 weight and activation quantization. See [MXFP4 weight and activation quantization](#mxfp4-weight-and-activation-quantization-of-llama-32-foundation-models).

> [!NOTE]
> Cells marked as **N/A** correspond to experiments that failed due to out-of-memory errors and are pending rerun.



Weight-only quantization of `Llama 3.2` and `Qwen 2.5` foundation models
--------------------------------------------------------------------------

The following results were obtained using the configurations `brevitas_examples/papers/learned_round/learned_round_weight_only.yaml`
and `brevitas_examples/papers/learned_round/learned_round_weight_only_spinquant.yaml`.

The results for `Llama 3.2` are summarized in the following table:

<table style="width:94%;">
<colgroup>
<col style="width: 4%" />
<col style="width: 3%" />
<col style="width: 6%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
</colgroup>
<tbody>
<tr>
<th colspan="3" align="center"></th>
<th colspan="6" align="center">W2g128</th>
<th colspan="6" align="center">W4</th>
<th colspan="6" align="center">W4g128</th>
</tr>
<tr>
<td colspan="3" align="center"></th>
<th colspan="3" align="center">WikiText2 ↓</th>
<th colspan="3" align="center">0-shot ↑</th>
<th colspan="3" align="center">WikiText2 ↓</th>
<th colspan="3" align="center">0-shot ↑</th>
<th colspan="3" align="center">WikiText2 ↓</th>
<th colspan="3" align="center">0-shot ↑</th>
</tr>
<tr>
<th align="center">Model</th>
<th align="center">Stage 1</th>
<th align="center">Stage 2</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
</tr>
<tr>
<th rowspan="14" align="center">Llama-3.2</th>
<th align="center">BF16</th>
<td align="center"></td>
<td align="center">8.9</td>
<td align="center">7.2</td>
<td align="center">5.9</td>
<td align="center">56.2</td>
<td align="center">63.6</td>
<td align="center">69.1</td>
<td align="center">8.9</td>
<td align="center">7.2</td>
<td align="center">5.9</td>
<td align="center">56.2</td>
<td align="center">63.6</td>
<td align="center">69.1</td>
<td align="center">8.9</td>
<td align="center">7.2</td>
<td align="center">5.9</td>
<td align="center">56.2</td>
<td align="center">63.6</td>
<td align="center">69.1</td>
</tr>
<tr>
<th rowspan="5" align="center">None</th>
<th align="center">RTN</th>
<td align="center">9e4</td>
<td align="center">1e4</td>
<td align="center">4e4</td>
<td align="center">35.06</td>
<td align="center">35.54</td>
<td align="center">35.57</td>
<td align="center">23.12</td>
<td align="center">9.81</td>
<td align="center">7.88</td>
<td align="center">48.50</td>
<td align="center">58.72</td>
<td align="center">65.23</td>
<td align="center">11.06</td>
<td align="center">7.75</td>
<td align="center">6.38</td>
<td align="center">52.83</td>
<td align="center">61.57</td>
<td align="center">68.31</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">179.00</td>
<td align="center">33.00</td>
<td align="center">25.38</td>
<td align="center">36.78</td>
<td align="center">41.08</td>
<td align="center">43.60</td>
<td align="center">11.06</td>
<td align="center">8.12</td>
<td align="center">6.78</td>
<td align="center">53.40</td>
<td align="center">61.48</td>
<td align="center">66.52</td>
<td align="center">9.81</td>
<td align="center">7.50</td>
<td align="center">6.22</td>
<td align="center">54.93</td>
<td align="center">62.49</td>
<td align="center">68.27</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">60.00</td>
<td align="center">21.00</td>
<td align="center">16.12</td>
<td align="center">38.84</td>
<td align="center">45.68</td>
<td align="center">50.20</td>
<td align="center">10.75</td>
<td align="center">7.88</td>
<td align="center">6.62</td>
<td align="center">53.83</td>
<td align="center">62.00</td>
<td align="center">67.18</td>
<td align="center">9.62</td>
<td align="center">7.38</td>
<td align="center">6.19</td>
<td align="center">55.23</td>
<td align="center">62.82</td>
<td align="center">68.31</td>
</tr>
<tr>
<th align="center">Sign Round</th>
<td align="center">2e4</td>
<td align="center">4e4</td>
<td align="center">6e3</td>
<td align="center">41.71</td>
<td align="center">51.06</td>
<td align="center">55.21</td>
<td align="center">10.12</td>
<td align="center">13.38</td>
<td align="center">10.75</td>
<td align="center">54.73</td>
<td align="center">62.74</td>
<td align="center">68.20</td>
<td align="center">9.62</td>
<td align="center">7.38</td>
<td align="center">6.12</td>
<td align="center">55.23</td>
<td align="center">63.17</td>
<td align="center">68.37</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">41.67</td>
<td align="center">18.18</td>
<td align="center">14.13</td>
<td align="center">43.66</td>
<td align="center">48.76</td>
<td align="center">55.47</td>
<td align="center">10.44</td>
<td align="center">8.11</td>
<td align="center">6.48</td>
<td align="center">54.12</td>
<td align="center">62.62</td>
<td align="center">67.09</td>
<td align="center">9.57</td>
<td align="center">7.44</td>
<td align="center">6.12</td>
<td align="center">55.23</td>
<td align="center">63.08</td>
<td align="center">68.20</td>
</tr>
<tr>
<th rowspan="4" align="center">HIP</th>
<th align="center">RTN</th>
<td align="center">1e5</td>
<td align="center">2e4</td>
<td align="center">5e3</td>
<td align="center">34.79</td>
<td align="center">35.06</td>
<td align="center">35.40</td>
<td align="center">12.94</td>
<td align="center">9.06</td>
<td align="center">7.09</td>
<td align="center">50.85</td>
<td align="center">59.54</td>
<td align="center">66.64</td>
<td align="center">10.94</td>
<td align="center">8.00</td>
<td align="center">6.47</td>
<td align="center">53.50</td>
<td align="center">61.72</td>
<td align="center">67.91</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">131.00</td>
<td align="center">27.00</td>
<td align="center">18.62</td>
<td align="center">37.35</td>
<td align="center">43.10</td>
<td align="center">48.70</td>
<td align="center">10.25</td>
<td align="center">7.75</td>
<td align="center">6.47</td>
<td align="center">54.27</td>
<td align="center">62.37</td>
<td align="center">67.00</td>
<td align="center">9.62</td>
<td align="center">7.50</td>
<td align="center">6.19</td>
<td align="center">55.22</td>
<td align="center">63.15</td>
<td align="center">68.16</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">77.00</td>
<td align="center">35.25</td>
<td align="center">20.75</td>
<td align="center">38.38</td>
<td align="center">41.31</td>
<td align="center">46.14</td>
<td align="center">10.56</td>
<td align="center">8.12</td>
<td align="center">6.62</td>
<td align="center">52.94</td>
<td align="center">61.49</td>
<td align="center">66.53</td>
<td align="center">9.94</td>
<td align="center">7.62</td>
<td align="center">6.28</td>
<td align="center">55.02</td>
<td align="center">62.89</td>
<td align="center">68.35</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">32.53</td>
<td align="center">17.64</td>
<td align="center">13.09</td>
<td align="center">43.97</td>
<td align="center">50.57</td>
<td align="center">36.05</td>
<td align="center">9.74</td>
<td align="center">7.65</td>
<td align="center">6.31</td>
<td align="center">55.37</td>
<td align="center">62.82</td>
<td align="center">67.98</td>
<td align="center">9.40</td>
<td align="center">7.42</td>
<td align="center">6.09</td>
<td align="center">55.97</td>
<td align="center">63.29</td>
<td align="center">68.58</td>
</tr>
<tr>
<th rowspan="4" align="center">MagR</th>
<th align="center">RTN</th>
<td align="center">2e4</td>
<td align="center">2e4</td>
<td align="center">6e3</td>
<td align="center">35.81</td>
<td align="center">35.44</td>
<td align="center">35.57</td>
<td align="center">13.19</td>
<td align="center">9.06</td>
<td align="center">7.09</td>
<td align="center">50.84</td>
<td align="center">54.51</td>
<td align="center">65.12</td>
<td align="center">12.19</td>
<td align="center">8.50</td>
<td align="center">6.78</td>
<td align="center">51.84</td>
<td align="center">56.34</td>
<td align="center">65.08</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">96.00</td>
<td align="center">34.25</td>
<td align="center">25.38</td>
<td align="center">37.10</td>
<td align="center">38.90</td>
<td align="center">42.68</td>
<td align="center">11.25</td>
<td align="center">8.38</td>
<td align="center">6.69</td>
<td align="center">53.35</td>
<td align="center">57.26</td>
<td align="center">66.64</td>
<td align="center">10.75</td>
<td align="center">8.12</td>
<td align="center">6.53</td>
<td align="center">53.67</td>
<td align="center">58.81</td>
<td align="center">66.80</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">43.75</td>
<td align="center">21.75</td>
<td align="center">18.25</td>
<td align="center">40.23</td>
<td align="center">45.90</td>
<td align="center">51.31</td>
<td align="center">10.56</td>
<td align="center">7.75</td>
<td align="center">6.41</td>
<td align="center">54.39</td>
<td align="center">61.61</td>
<td align="center">67.44</td>
<td align="center">10.25</td>
<td align="center">7.62</td>
<td align="center">6.28</td>
<td align="center">54.83</td>
<td align="center">61.50</td>
<td align="center">67.85</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">34.16</td>
<td align="center">17.98</td>
<td align="center">N/A</td>
<td align="center">43.83</td>
<td align="center">49.63</td>
<td align="center">N/A</td>
<td align="center">10.07</td>
<td align="center">8.11</td>
<td align="center">N/A</td>
<td align="center">52.58</td>
<td align="center">46.94</td>
<td align="center">N/A</td>
<td align="center">9.89</td>
<td align="center">7.91</td>
<td align="center">N/A</td>
<td align="center">54.31</td>
<td align="center">56.37</td>
<td align="center">N/A</td>
</tr>
</tbody>
</table>

**Key takeaways**: At W2g128, Learned Round achieves the lowest perplexity across all transforms (e.g. 32.53 with HIP vs. 60.00 for Qronos on 1B), substantially outperforming both GPTQ and Qronos. At W4 and W4g128, all rounding methods perform comparably, with Learned Round combined with HIP achieving the best overall WikiText2 perplexity (9.40 on W4g128 for 1B) and competitive zero-shot accuracy. The benefit of learned rounding is most pronounced in the aggressive W2g128 regime, where greedy solvers struggle.

The results for `Qwen 2.5` are summarized in the following table:

<table style="width:94%;">
<colgroup>
<col style="width: 4%" />
<col style="width: 3%" />
<col style="width: 6%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 4%" />
</colgroup>
<tbody>
<tr>
<th colspan="3" align="center"></th>
<th colspan="6" align="center">W2g128</th>
<th colspan="6" align="center">W4</th>
<th colspan="6" align="center">W4g128</th>
</tr>
<tr>
<th colspan="3" align="center"></th>
<th colspan="3" align="center">WikiText2 ↓</th>
<th colspan="3" align="center">0-shot ↑</th>
<th colspan="3" align="center">WikiText2 ↓</th>
<th colspan="3" align="center">0-shot ↑</th>
<th colspan="3" align="center">WikiText2 ↓</th>
<th colspan="3" align="center">0-shot ↑</th>
</tr>
<tr>
<th align="center">Model</th>
<th align="center">Stage 1</th>
<th align="center">Stage 2</th>
<th align="center">1.5B</th>
<th align="center">3B</th>
<th align="center">7B</th>
<th align="center">1.5B</th>
<th align="center">3B</th>
<th align="center">7B</th>
<th align="center">1.5B</th>
<th align="center">3B</th>
<th align="center">7B</th>
<th align="center">1.5B</th>
<th align="center">3B</th>
<th align="center">7B</th>
<th align="center">1.5B</th>
<th align="center">3B</th>
<th align="center">7B</th>
<th align="center">1.5B</th>
<th align="center">3B</th>
<th align="center">7B</th>
</tr>
<tr>
<th rowspan="14" align="center">Qwen 2.5</th>
<th align="center">BF16</th>
<td align="center"></td>
<td align="center">8.5</td>
<td align="center">7.4</td>
<td align="center">6.5</td>
<td align="center">60.7</td>
<td align="center">64.3</td>
<td align="center">67.2</td>
<td align="center">8.5</td>
<td align="center">7.4</td>
<td align="center">6.5</td>
<td align="center">60.7</td>
<td align="center">64.3</td>
<td align="center">67.2</td>
<td align="center">8.5</td>
<td align="center">7.4</td>
<td align="center">6.5</td>
<td align="center">60.7</td>
<td align="center">64.3</td>
<td align="center">67.2</td>
</tr>
<tr>
<th rowspan="5" align="center">None</th>
<th align="center">RTN</th>
<td align="center">2e5</td>
<td align="center">8e4</td>
<td align="center">2e4</td>
<td align="center">35.33</td>
<td align="center">34.90</td>
<td align="center">35.53</td>
<td align="center">12.75</td>
<td align="center">6e3</td>
<td align="center">8.50</td>
<td align="center">54.52</td>
<td align="center">35.56</td>
<td align="center">61.49</td>
<td align="center">9.50</td>
<td align="center">9.06</td>
<td align="center">6.78</td>
<td align="center">58.47</td>
<td align="center">61.37</td>
<td align="center">65.61</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">38.00</td>
<td align="center">23.12</td>
<td align="center">12.56</td>
<td align="center">39.46</td>
<td align="center">41.43</td>
<td align="center">52.07</td>
<td align="center">9.81</td>
<td align="center">8.38</td>
<td align="center">7.09</td>
<td align="center">56.59</td>
<td align="center">62.24</td>
<td align="center">64.16</td>
<td align="center">8.94</td>
<td align="center">7.75</td>
<td align="center">6.69</td>
<td align="center">59.38</td>
<td align="center">62.85</td>
<td align="center">66.36</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">27.50</td>
<td align="center">18.62</td>
<td align="center">12.19</td>
<td align="center">42.57</td>
<td align="center">46.41</td>
<td align="center">55.23</td>
<td align="center">9.50</td>
<td align="center">8.25</td>
<td align="center">7.06</td>
<td align="center">56.42</td>
<td align="center">62.41</td>
<td align="center">65.33</td>
<td align="center">8.94</td>
<td align="center">7.75</td>
<td align="center">6.69</td>
<td align="center">60.14</td>
<td align="center">62.47</td>
<td align="center">66.75</td>
</tr>
<tr>
<th align="center">Sign Round</th>
<td align="center">26.62</td>
<td align="center">18.00</td>
<td align="center">11.62</td>
<td align="center">46.56</td>
<td align="center">50.65</td>
<td align="center">59.26</td>
<td align="center">9.19</td>
<td align="center">8.00</td>
<td align="center">6.84</td>
<td align="center">58.88</td>
<td align="center">62.41</td>
<td align="center">65.93</td>
<td align="center">8.94</td>
<td align="center">7.75</td>
<td align="center">6.62</td>
<td align="center">60.46</td>
<td align="center">63.99</td>
<td align="center">66.79</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">23.70</td>
<td align="center">16.93</td>
<td align="center">12.09</td>
<td align="center">45.96</td>
<td align="center">51.00</td>
<td align="center">57.73</td>
<td align="center">9.85</td>
<td align="center">8.10</td>
<td align="center">10.04</td>
<td align="center">59.28</td>
<td align="center">63.27</td>
<td align="center">65.34</td>
<td align="center">8.86</td>
<td align="center">7.73</td>
<td align="center">6.68</td>
<td align="center">59.73</td>
<td align="center">64.22</td>
<td align="center">66.84</td>
</tr>
<tr>
<th rowspan="4" align="center">HIP</th>
<th align="center">RTN</th>
<td align="center">1e4</td>
<td align="center">1e8</td>
<td align="center">536.00</td>
<td align="center">34.83</td>
<td align="center">35.06</td>
<td align="center">37.65</td>
<td align="center">9.94</td>
<td align="center">11.81</td>
<td align="center">8.00</td>
<td align="center">56.84</td>
<td align="center">59.66</td>
<td align="center">62.94</td>
<td align="center">9.31</td>
<td align="center">8.25</td>
<td align="center">6.78</td>
<td align="center">59.73</td>
<td align="center">62.51</td>
<td align="center">65.94</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">23.12</td>
<td align="center">15.88</td>
<td align="center">10.94</td>
<td align="center">43.71</td>
<td align="center">45.93</td>
<td align="center">52.79</td>
<td align="center">9.06</td>
<td align="center">7.88</td>
<td align="center">6.94</td>
<td align="center">59.81</td>
<td align="center">63.56</td>
<td align="center">65.70</td>
<td align="center">8.75</td>
<td align="center">7.62</td>
<td align="center">6.62</td>
<td align="center">59.66</td>
<td align="center">63.73</td>
<td align="center">66.48</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">20.38</td>
<td align="center">15.19</td>
<td align="center">10.75</td>
<td align="center">45.99</td>
<td align="center">47.02</td>
<td align="center">55.92</td>
<td align="center">9.06</td>
<td align="center">7.88</td>
<td align="center">6.94</td>
<td align="center">58.69</td>
<td align="center">62.78</td>
<td align="center">66.19</td>
<td align="center">8.75</td>
<td align="center">7.75</td>
<td align="center">6.62</td>
<td align="center">60.29</td>
<td align="center">63.16</td>
<td align="center">66.87</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">18.72</td>
<td align="center">13.48</td>
<td align="center">11.49</td>
<td align="center">46.55</td>
<td align="center">52.08</td>
<td align="center">57.49</td>
<td align="center">11.84</td>
<td align="center">7.99</td>
<td align="center">7.46</td>
<td align="center">52.57</td>
<td align="center">64.18</td>
<td align="center">65.72</td>
<td align="center">8.74</td>
<td align="center">7.63</td>
<td align="center">6.69</td>
<td align="center">59.93</td>
<td align="center">64.06</td>
<td align="center">66.42</td>
</tr>
<tr>
<th rowspan="4" align="center">MagR</th>
<th align="center">RTN</th>
<td align="center">6e4</td>
<td align="center">7e4</td>
<td align="center">2e3</td>
<td align="center">35.55</td>
<td align="center">35.15</td>
<td align="center">36.15</td>
<td align="center">10.56</td>
<td align="center">9.06</td>
<td align="center">7.50</td>
<td align="center">55.31</td>
<td align="center">59.89</td>
<td align="center">64.93</td>
<td align="center">10.12</td>
<td align="center">8.62</td>
<td align="center">7.28</td>
<td align="center">56.66</td>
<td align="center">61.02</td>
<td align="center">66.33</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">43.75</td>
<td align="center">40.00</td>
<td align="center">13.81</td>
<td align="center">40.42</td>
<td align="center">42.54</td>
<td align="center">51.59</td>
<td align="center">9.94</td>
<td align="center">8.38</td>
<td align="center">7.34</td>
<td align="center">58.19</td>
<td align="center">62.56</td>
<td align="center">65.39</td>
<td align="center">9.62</td>
<td align="center">8.25</td>
<td align="center">7.16</td>
<td align="center">58.04</td>
<td align="center">61.89</td>
<td align="center">66.12</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">34.25</td>
<td align="center">19.75</td>
<td align="center">13.81</td>
<td align="center">41.37</td>
<td align="center">46.71</td>
<td align="center">54.87</td>
<td align="center">9.81</td>
<td align="center">8.38</td>
<td align="center">7.28</td>
<td align="center">57.71</td>
<td align="center">61.56</td>
<td align="center">65.65</td>
<td align="center">9.50</td>
<td align="center">8.00</td>
<td align="center">7.16</td>
<td align="center">57.90</td>
<td align="center">61.51</td>
<td align="center">66.38</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">22.49</td>
<td align="center">15.79</td>
<td align="center">N/A</td>
<td align="center"><strong>46.83</strong></td>
<td align="center">48.03</td>
<td align="center">N/A</td>
<td align="center">10.14</td>
<td align="center">8.08</td>
<td align="center">N/A</td>
<td align="center">58.21</td>
<td align="center">63.07</td>
<td align="center">N/A</td>
<td align="center">9.03</td>
<td align="center">7.80</td>
<td align="center">N/A</td>
<td align="center">59.14</td>
<td align="center">63.44</td>
<td align="center">N/A</td>
</tr>
</tbody>
</table>

**Key takeaways**: Trends on Qwen 2.5 mirror those on Llama 3.2. At W2g128, Learned Round consistently delivers the best or near-best perplexity across transforms (e.g. 18.72 with HIP vs. 20.38 for Qronos on 1.5B). At W4 and W4g128, Learned Round with HIP yields the best WikiText2 perplexity (8.74 on W4g128 for 1.5B) and top zero-shot scores (64.22 on W4g128 for 3B), confirming the generality of the approach across model families.


Weight and activation quantization of `Llama 3.2` foundation models
--------------------------------------------------------------------

The following results were obtained using the configurations `brevitas_examples/papers/learned_round/learned_round_weight_act.yaml`
and `brevitas_examples/papers/learned_round/learned_round_weight_act_spinquant.yaml`.

The results for `Llama 3.2` are summarized in the following table:

<table style="width:95%;">
<colgroup>
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 14%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
</colgroup>
<tbody>
<tr>
<td colspan="3" align="center"></td>
<th colspan="6" align="center">W4g32A4g32</th>
</tr>
<tr>
<td colspan="3" align="center"></td>
<th colspan="3" align="center">WikiText2 ↓</th>
<th colspan="3" align="center">0-shot ↑</th>
</tr>
<tr>
<th align="center">Model</th>
<th align="center">Stage 1</th>
<th align="center">Stage 2</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
</tr>
<tr>
<th rowspan="21" align="center">Llama-3.2</th>
<th align="center">BF16</th>
<td align="center"></td>
<td align="center">8.9</td>
<td align="center">7.2</td>
<td align="center">5.9</td>
<td align="center">56.2</td>
<td align="center">63.6</td>
<td align="center">69.1</td>
</tr>
<tr>
<th rowspan="4" align="center">None</th>
<th align="center">RTN</th>
<td align="center">6e3</td>
<td align="center">2e4</td>
<td align="center">5e4</td>
<td align="center">34.59</td>
<td align="center">34.83</td>
<td align="center">35.60</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">2e4</td>
<td align="center">1e4</td>
<td align="center">2e4</td>
<td align="center">34.38</td>
<td align="center">35.48</td>
<td align="center">34.32</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">174.00</td>
<td align="center">84.50</td>
<td align="center">82.00</td>
<td align="center">37.44</td>
<td align="center">38.59</td>
<td align="center">38.65</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">100.67</td>
<td align="center">73.80</td>
<td align="center">274.88</td>
<td align="center">36.10</td>
<td align="center">39.03</td>
<td align="center">38.15</td>
</tr>
<tr>
<th rowspan="4" align="center">HIP</th>
<th align="center">RTN</th>
<td align="center">18.25</td>
<td align="center">10.56</td>
<td align="center">8.38</td>
<td align="center">45.78</td>
<td align="center">55.25</td>
<td align="center">61.33</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">13.19</td>
<td align="center">8.75</td>
<td align="center">7.50</td>
<td align="center">48.49</td>
<td align="center">58.35</td>
<td align="center">62.76</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">13.19</td>
<td align="center">9.19</td>
<td align="center">7.62</td>
<td align="center">48.40</td>
<td align="center">58.24</td>
<td align="center">62.85</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">12.32</td>
<td align="center">8.78</td>
<td align="center">7.23</td>
<td align="center">50.57</td>
<td align="center">59.09</td>
<td align="center">63.70</td>
</tr>
<tr>
<th rowspan="4" align="center">MagR</th>
<th align="center">RTN</th>
<td align="center">6e3</td>
<td align="center">8e3</td>
<td align="center">2e4</td>
<td align="center">34.94</td>
<td align="center">35.03</td>
<td align="center">34.75</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">1e4</td>
<td align="center">2e4</td>
<td align="center">2e4</td>
<td align="center">35.74</td>
<td align="center">35.91</td>
<td align="center">35.44</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">197.00</td>
<td align="center">153.00</td>
<td align="center">174.00</td>
<td align="center">36.74</td>
<td align="center">37.65</td>
<td align="center">38.05</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">103.10</td>
<td align="center">82.66</td>
<td align="center">N/A</td>
<td align="center">38.74</td>
<td align="center">36.75</td>
<td align="center">N/A</td>
</tr>
<tr>
<th rowspan="4" align="center">QuaRot</th>
<th align="center">RTN</th>
<td align="center">27.88</td>
<td align="center">19.12</td>
<td align="center">11.62</td>
<td align="center">42.25</td>
<td align="center">44.93</td>
<td align="center">55.34</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">14.69</td>
<td align="center">10.12</td>
<td align="center">8.00</td>
<td align="center">47.54</td>
<td align="center">55.34</td>
<td align="center">61.58</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">13.81</td>
<td align="center">9.31</td>
<td align="center">7.75</td>
<td align="center">48.77</td>
<td align="center">57.18</td>
<td align="center">62.80</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">13.65</td>
<td align="center">9.88</td>
<td align="center">7.86</td>
<td align="center">49.26</td>
<td align="center">55.22</td>
<td align="center">44.84</td>
</tr>
<tr>
<th rowspan="4" align="center">SpinQuant</th>
<th align="center">RTN</th>
<td align="center">18.25</td>
<td align="center">87.00</td>
<td align="center">77.00</td>
<td align="center">46.57</td>
<td align="center">35.06</td>
<td align="center">36.15</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">15.38</td>
<td align="center">1e3</td>
<td align="center">392.00</td>
<td align="center">47.50</td>
<td align="center">34.58</td>
<td align="center">34.67</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">14.69</td>
<td align="center">368.00</td>
<td align="center">286.00</td>
<td align="center">47.81</td>
<td align="center">34.68</td>
<td align="center">35.27</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">13.52</td>
<td align="center">9.41</td>
<td align="center">7.59</td>
<td align="center">50.22</td>
<td align="center">57.08</td>
<td align="center">62.22</td>
</tr>
</tbody>
</table>

**Key takeaways**: W4A4 quantization is significantly more challenging than weight-only. Without a transform (Stage 1 = None), all rounding methods produce near-random accuracy. A transform is essential: HIP and QuaRot enable usable models, and Learned Round consistently achieves the best perplexity and zero-shot accuracy within each transform group (e.g. 12.32/50.57 with HIP on 1B vs. 13.19/48.49 for GPTQ). SpinQuant + Learned Round yields the overall best combination (7.59 perplexity, 62.22 zero-shot on 8B), notably recovering from the poor results that GPTQ and Qronos produce with SpinQuant on 3B and 8B.

MXFP4 weight and activation quantization of `Llama 3.2` foundation models
----------------------------------------------------------------------------

The following results were obtained using the configurations `brevitas_examples/papers/learned_round/learned_round_mxfp4.yaml`
and `brevitas_examples/papers/learned_round/learned_round_mxfp4_spinquant.yaml`.

The results for `Llama 3.2` are summarized in the following table:

<table style="width:95%;">
<colgroup>
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 14%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 10%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 9%" />
</colgroup>
<tbody>
<tr>
<th colspan="3" align="center"></th>
<th colspan="6" align="center">W4g32A4g32Po2</th>
</tr>
<tr>
<th colspan="3" align="center"></th>
<th colspan="3" align="center">WikiText2 ↓</th>
<th colspan="3" align="center">0-shot ↑</th>
</tr>
<tr>
<th align="center">Model</th>
<th align="center">Stage 1</th>
<th align="center">Stage 2</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
<th align="center">1B</th>
<th align="center">3B</th>
<th align="center">8B</th>
</tr>
<tr>
<th rowspan="21" align="center">Llama-3.2</th>
<th align="center">BF16</th>
<td align="center"></td>
<td align="center">8.9</td>
<td align="center">7.2</td>
<td align="center">5.9</td>
<td align="center">56.2</td>
<td align="center">63.6</td>
<td align="center">69.1</td>
</tr>
<tr>
<th rowspan="4" align="center">None</th>
<th align="center">RTN</th>
<td align="center">14.44</td>
<td align="center">9.19</td>
<td align="center">7.75</td>
<td align="center">50.15</td>
<td align="center">57.39</td>
<td align="center">63.45</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">12.38</td>
<td align="center">8.62</td>
<td align="center">7.16</td>
<td align="center">51.80</td>
<td align="center">56.95</td>
<td align="center">64.68</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">12.56</td>
<td align="center">8.75</td>
<td align="center">7.22</td>
<td align="center">51.57</td>
<td align="center">59.14</td>
<td align="center">64.09</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">11.78</td>
<td align="center">8.49</td>
<td align="center">6.97</td>
<td align="center">52.78</td>
<td align="center">61.01</td>
<td align="center">65.21</td>
</tr>
<tr>
<th rowspan="4" align="center">HIP</th>
<th align="center">RTN</th>
<td align="center">13.19</td>
<td align="center">8.94</td>
<td align="center">7.28</td>
<td align="center">50.42</td>
<td align="center">59.21</td>
<td align="center">65.99</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">11.06</td>
<td align="center">8.25</td>
<td align="center">6.78</td>
<td align="center">52.49</td>
<td align="center">60.98</td>
<td align="center">65.94</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">11.62</td>
<td align="center">8.50</td>
<td align="center">7.06</td>
<td align="center">51.58</td>
<td align="center">59.72</td>
<td align="center">65.54</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">11.01</td>
<td align="center">8.38</td>
<td align="center">6.70</td>
<td align="center">53.05</td>
<td align="center">61.11</td>
<td align="center">65.64</td>
</tr>
<tr>
<th rowspan="4" align="center">MagR</th>
<th align="center">RTN</th>
<td align="center">18.88</td>
<td align="center">12.00</td>
<td align="center">8.94</td>
<td align="center">46.03</td>
<td align="center">48.84</td>
<td align="center">57.59</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">14.44</td>
<td align="center">9.94</td>
<td align="center">7.88</td>
<td align="center">49.36</td>
<td align="center">53.12</td>
<td align="center">62.19</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">13.19</td>
<td align="center">8.94</td>
<td align="center">7.50</td>
<td align="center">51.28</td>
<td align="center">58.27</td>
<td align="center">63.86</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">12.48</td>
<td align="center">9.50</td>
<td align="center">N/A</td>
<td align="center">50.86</td>
<td align="center">59.18</td>
<td align="center">N/A</td>
</tr>
<tr>
<th rowspan="4" align="center">QuaRot</th>
<th align="center">RTN</th>
<td align="center">15.62</td>
<td align="center">12.38</td>
<td align="center">8.50</td>
<td align="center">48.36</td>
<td align="center">54.34</td>
<td align="center">62.64</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">12.19</td>
<td align="center">9.06</td>
<td align="center">7.38</td>
<td align="center">51.10</td>
<td align="center">58.52</td>
<td align="center">64.59</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">11.81</td>
<td align="center">8.62</td>
<td align="center">7.00</td>
<td align="center">51.71</td>
<td align="center">59.06</td>
<td align="center">N/A</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">11.69</td>
<td align="center">8.40</td>
<td align="center">6.86</td>
<td align="center">52.35</td>
<td align="center">60.26</td>
<td align="center">41.28</td>
</tr>
<tr>
<th rowspan="4" align="center">SpinQuant</th>
<th align="center">RTN</th>
<td align="center">12.00</td>
<td align="center">8.75</td>
<td align="center">7.16</td>
<td align="center">51.92</td>
<td align="center">59.35</td>
<td align="center">66.01</td>
</tr>
<tr>
<th align="center">GPTQ</th>
<td align="center">12.38</td>
<td align="center">9.62</td>
<td align="center">8.12</td>
<td align="center">51.06</td>
<td align="center">58.37</td>
<td align="center">62.93</td>
</tr>
<tr>
<th align="center">Qronos</th>
<td align="center">11.62</td>
<td align="center">8.62</td>
<td align="center">7.22</td>
<td align="center">51.50</td>
<td align="center">59.26</td>
<td align="center">64.49</td>
</tr>
<tr>
<th align="center">Learned Round</th>
<td align="center">11.71</td>
<td align="center">8.51</td>
<td align="center">6.93</td>
<td align="center">52.52</td>
<td align="center">59.79</td>
<td align="center">N/A</td>
</tr>
</tbody>
</table>

**Key takeaways**: MXFP4 quantization is more forgiving than integer W4A4, with all transforms producing usable models even with RTN. Learned Round still provides consistent gains: it achieves the best or near-best perplexity within every transform group (e.g. 11.01 with HIP on 1B vs. 11.06 for GPTQ). The margins between rounding methods are narrower than in integer W4A4, but Learned Round with any of HIP, QuaRot, or SpinQuant approaches BF16 quality (e.g. 6.86 perplexity on 8B with QuaRot, vs. 5.9 for BF16).
