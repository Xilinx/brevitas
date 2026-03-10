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
ts performance was compared against the [SignRound](https://aclanthology.org/2024.findings-emnlp.662.pdf) for weight-only quantization,
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
<th colspan="3"></th>
<th colspan="6">W2g128</th>
<th colspan="6">W4</th>
<th colspan="6">W4g128</th>
</tr>
<tr>
<td colspan="3"></th>
<th colspan="3">WikiText2 ↓</th>
<th colspan="3">0-shot ↑</th>
<th colspan="3">WikiText2 ↓</th>
<th colspan="3">0-shot ↑</th>
<th colspan="3">WikiText2 ↓</th>
<th colspan="3">0-shot ↑</th>
</tr>
<tr>
<th>Model</th>
<th>Stage 1</th>
<th>Stage 2</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
</tr>
<tr>
<th rowspan="14">Llama-3.2</th>
<th>BF16</th>
<td></td>
<td>8.9</td>
<td>7.2</td>
<td>5.9</td>
<td>56.2</td>
<td>63.6</td>
<td>69.1</td>
<td>8.9</td>
<td>7.2</td>
<td>5.9</td>
<td>56.2</td>
<td>63.6</td>
<td>69.1</td>
<td>8.9</td>
<td>7.2</td>
<td>5.9</td>
<td>56.2</td>
<td>63.6</td>
<td>69.1</td>
</tr>
<tr>
<th rowspan="5">None</th>
<th>RTN</th>
<td>92672.00</td>
<td>11776.00</td>
<td>38656.00</td>
<td>35.06</td>
<td>35.54</td>
<td>35.57</td>
<td>23.12</td>
<td>9.81</td>
<td>7.88</td>
<td>48.50</td>
<td>58.72</td>
<td>65.23</td>
<td>11.06</td>
<td>7.75</td>
<td>6.38</td>
<td>52.83</td>
<td>61.57</td>
<td>68.31</td>
</tr>
<tr>
<th>GPTQ</th>
<td>179.00</td>
<td>33.00</td>
<td>25.38</td>
<td>36.78</td>
<td>41.08</td>
<td>43.60</td>
<td>11.06</td>
<td>8.12</td>
<td>6.78</td>
<td>53.40</td>
<td>61.48</td>
<td>66.52</td>
<td>9.81</td>
<td>7.50</td>
<td>6.22</td>
<td>54.93</td>
<td>62.49</td>
<td>68.27</td>
</tr>
<tr>
<th>Qronos</th>
<td>60.00</td>
<td>21.00</td>
<td>16.12</td>
<td>38.84</td>
<td>45.68</td>
<td>50.20</td>
<td>10.75</td>
<td>7.88</td>
<td>6.62</td>
<td>53.83</td>
<td>62.00</td>
<td>67.18</td>
<td>9.62</td>
<td>7.38</td>
<td>6.19</td>
<td>55.23</td>
<td>62.82</td>
<td>68.31</td>
</tr>
<tr>
<th>Sign Round</th>
<td>17151.00</td>
<td>36352.00</td>
<td>6304.00</td>
<td>41.71</td>
<td>51.06</td>
<td>55.21</td>
<td>10.12</td>
<td>13.38</td>
<td>10.75</td>
<td>54.73</td>
<td>62.74</td>
<td>68.20</td>
<td>9.62</td>
<td>7.38</td>
<td>6.12</td>
<td>55.23</td>
<td>63.17</td>
<td>68.37</td>
</tr>
<tr>
<th>Learned Round</th>
<td>41.67</td>
<td>18.18</td>
<td>14.13</td>
<td>43.66</td>
<td>48.76</td>
<td>55.47</td>
<td>10.44</td>
<td>8.11</td>
<td>6.48</td>
<td>54.12</td>
<td>62.62</td>
<td>67.09</td>
<td>9.57</td>
<td>7.44</td>
<td>6.12</td>
<td>55.23</td>
<td>63.08</td>
<td>68.20</td>
</tr>
<tr>
<th rowspan="4">HIP</th>
<th>RTN</th>
<td>143360.00</td>
<td>16128.00</td>
<td>4928.00</td>
<td>34.79</td>
<td>35.06</td>
<td>35.40</td>
<td>12.94</td>
<td>9.06</td>
<td>7.09</td>
<td>50.85</td>
<td>59.54</td>
<td>66.64</td>
<td>10.94</td>
<td>8.00</td>
<td>6.47</td>
<td>53.50</td>
<td>61.72</td>
<td>67.91</td>
</tr>
<tr>
<th>GPTQ</th>
<td>131.00</td>
<td>27.00</td>
<td>18.62</td>
<td>37.35</td>
<td>43.10</td>
<td>48.70</td>
<td>10.25</td>
<td>7.75</td>
<td>6.47</td>
<td>54.27</td>
<td>62.37</td>
<td>67.00</td>
<td>9.62</td>
<td>7.50</td>
<td>6.19</td>
<td>55.22</td>
<td>63.15</td>
<td>68.16</td>
</tr>
<tr>
<th>Qronos</th>
<td>77.00</td>
<td>35.25</td>
<td>20.75</td>
<td>38.38</td>
<td>41.31</td>
<td>46.14</td>
<td>10.56</td>
<td>8.12</td>
<td>6.62</td>
<td>52.94</td>
<td>61.49</td>
<td>66.53</td>
<td>9.94</td>
<td>7.62</td>
<td>6.28</td>
<td>55.02</td>
<td>62.89</td>
<td>68.35</td>
</tr>
<tr>
<th>Learned Round</th>
<td>32.53</td>
<td>17.64</td>
<td>13.09</td>
<td>43.97</td>
<td>50.57</td>
<td>36.05</td>
<td>9.74</td>
<td>7.65</td>
<td>6.31</td>
<td>55.37</td>
<td>62.82</td>
<td>67.98</td>
<td>9.40</td>
<td>7.42</td>
<td>6.09</td>
<td>55.97</td>
<td>63.29</td>
<td>68.58</td>
</tr>
<tr>
<th rowspan="4">MagR</th>
<th>RTN</th>
<td>20736.00</td>
<td>16128.00</td>
<td>5568.00</td>
<td>35.81</td>
<td>35.44</td>
<td>35.57</td>
<td>13.19</td>
<td>9.06</td>
<td>7.09</td>
<td>50.84</td>
<td>54.51</td>
<td>65.12</td>
<td>12.19</td>
<td>8.50</td>
<td>6.78</td>
<td>51.84</td>
<td>56.34</td>
<td>65.08</td>
</tr>
<tr>
<th>GPTQ</th>
<td>96.00</td>
<td>34.25</td>
<td>25.38</td>
<td>37.10</td>
<td>38.90</td>
<td>42.68</td>
<td>11.25</td>
<td>8.38</td>
<td>6.69</td>
<td>53.35</td>
<td>57.26</td>
<td>66.64</td>
<td>10.75</td>
<td>8.12</td>
<td>6.53</td>
<td>53.67</td>
<td>58.81</td>
<td>66.80</td>
</tr>
<tr>
<th>Qronos</th>
<td>43.75</td>
<td>21.75</td>
<td>18.25</td>
<td>40.23</td>
<td>45.90</td>
<td>51.31</td>
<td>10.56</td>
<td>7.75</td>
<td>6.41</td>
<td>54.39</td>
<td>61.61</td>
<td>67.44</td>
<td>10.25</td>
<td>7.62</td>
<td>6.28</td>
<td>54.83</td>
<td>61.50</td>
<td>67.85</td>
</tr>
<tr>
<th>Learned Round</th>
<td>34.16</td>
<td>17.98</td>
<td></td>
<td>43.83</td>
<td>49.63</td>
<td></td>
<td>10.07</td>
<td>8.11</td>
<td></td>
<td>52.58</td>
<td>46.94</td>
<td></td>
<td>9.89</td>
<td>7.91</td>
<td></td>
<td>54.31</td>
<td>56.37</td>
<td></td>
</tr>
</tbody>
</table>

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
<th colspan="3"></th>
<th colspan="6">W2g128</th>
<th colspan="6">W4</th>
<th colspan="6">W4g128</th>
</tr>
<tr>
<th colspan="3"></th>
<th colspan="3">WikiText2 ↓</th>
<th colspan="3">0-shot ↑</th>
<th colspan="3">WikiText2 ↓</th>
<th colspan="3">0-shot ↑</th>
<th colspan="3">WikiText2 ↓</th>
<th colspan="3">0-shot ↑</th>
</tr>
<tr>
<th>Model</th>
<th>Stage 1</th>
<th>Stage 2</th>
<th>1.5B</th>
<th>3B</th>
<th>7B</th>
<th>1.5B</th>
<th>3B</th>
<th>7B</th>
<th>1.5B</th>
<th>3B</th>
<th>7B</th>
<th>1.5B</th>
<th>3B</th>
<th>7B</th>
<th>1.5B</th>
<th>3B</th>
<th>7B</th>
<th>1.5B</th>
<th>3B</th>
<th>7B</th>
</tr>
<tr>
<th rowspan="14">Qwen 2.5</th>
<th>BF16</th>
<td></td>
<td>8.5</td>
<td>7.4</td>
<td>6.5</td>
<td>60.7</td>
<td>64.3</td>
<td>67.2</td>
<td>8.5</td>
<td>7.4</td>
<td>6.5</td>
<td>60.7</td>
<td>64.3</td>
<td>67.2</td>
<td>8.5</td>
<td>7.4</td>
<td>6.5</td>
<td>60.7</td>
<td>64.3</td>
<td>67.2</td>
</tr>
<tr>
<th rowspan="5">None</th>
<th>RTN</th>
<td>152576.00</td>
<td>76800.00</td>
<td>19456.00</td>
<td>35.33</td>
<td>34.90</td>
<td>35.53</td>
<td>12.75</td>
<td>6304.00</td>
<td>8.50</td>
<td>54.52</td>
<td>35.56</td>
<td>61.49</td>
<td>9.50</td>
<td>9.06</td>
<td>6.78</td>
<td>58.47</td>
<td>61.37</td>
<td>65.61</td>
</tr>
<tr>
<th>GPTQ</th>
<td>38.00</td>
<td>23.12</td>
<td>12.56</td>
<td>39.46</td>
<td>41.43</td>
<td>52.07</td>
<td>9.81</td>
<td>8.38</td>
<td>7.09</td>
<td>56.59</td>
<td>62.24</td>
<td>64.16</td>
<td>8.94</td>
<td>7.75</td>
<td>6.69</td>
<td>59.38</td>
<td>62.85</td>
<td>66.36</td>
</tr>
<tr>
<th>Qronos</th>
<td>27.50</td>
<td>18.62</td>
<td>12.19</td>
<td>42.57</td>
<td>46.41</td>
<td>55.23</td>
<td>9.50</td>
<td>8.25</td>
<td>7.06</td>
<td>56.42</td>
<td>62.41</td>
<td>65.33</td>
<td>8.94</td>
<td>7.75</td>
<td>6.69</td>
<td>60.14</td>
<td>62.47</td>
<td>66.75</td>
</tr>
<tr>
<th>Sign Round</th>
<td>26.62</td>
<td>18.00</td>
<td>11.62</td>
<td>46.56</td>
<td>50.65</td>
<td>59.26</td>
<td>9.19</td>
<td>8.00</td>
<td>6.84</td>
<td>58.88</td>
<td>62.41</td>
<td>65.93</td>
<td>8.94</td>
<td>7.75</td>
<td>6.62</td>
<td>60.46</td>
<td>63.99</td>
<td>66.79</td>
</tr>
<tr>
<th>Learned Round</th>
<td>23.70</td>
<td>16.93</td>
<td>12.09</td>
<td>45.96</td>
<td>51.00</td>
<td>57.73</td>
<td>9.85</td>
<td>8.10</td>
<td>10.04</td>
<td>59.28</td>
<td>63.27</td>
<td>65.34</td>
<td>8.86</td>
<td>7.73</td>
<td>6.68</td>
<td>59.73</td>
<td>64.22</td>
<td>66.84</td>
</tr>
<tr>
<th rowspan="4">HIP</th>
<th>RTN</th>
<td>14208.00</td>
<td>95420416.00</td>
<td>536.00</td>
<td>34.83</td>
<td>35.06</td>
<td>37.65</td>
<td>9.94</td>
<td>11.81</td>
<td>8.00</td>
<td>56.84</td>
<td>59.66</td>
<td>62.94</td>
<td>9.31</td>
<td>8.25</td>
<td>6.78</td>
<td>59.73</td>
<td>62.51</td>
<td>65.94</td>
</tr>
<tr>
<th>GPTQ</th>
<td>23.12</td>
<td>15.88</td>
<td>10.94</td>
<td>43.71</td>
<td>45.93</td>
<td>52.79</td>
<td>9.06</td>
<td>7.88</td>
<td>6.94</td>
<td>59.81</td>
<td>63.56</td>
<td>65.70</td>
<td>8.75</td>
<td>7.62</td>
<td>6.62</td>
<td>59.66</td>
<td>63.73</td>
<td>66.48</td>
</tr>
<tr>
<th>Qronos</th>
<td>20.38</td>
<td>15.19</td>
<td>10.75</td>
<td>45.99</td>
<td>47.02</td>
<td>55.92</td>
<td>9.06</td>
<td>7.88</td>
<td>6.94</td>
<td>58.69</td>
<td>62.78</td>
<td>66.19</td>
<td>8.75</td>
<td>7.75</td>
<td>6.62</td>
<td>60.29</td>
<td>63.16</td>
<td>66.87</td>
</tr>
<tr>
<th>Learned Round</th>
<td>18.72</td>
<td>13.48</td>
<td>11.49</td>
<td>46.55</td>
<td>52.08</td>
<td>57.49</td>
<td>11.84</td>
<td>7.99</td>
<td>7.46</td>
<td>52.57</td>
<td>64.18</td>
<td>65.72</td>
<td>8.74</td>
<td>7.63</td>
<td>6.69</td>
<td>59.93</td>
<td>64.06</td>
<td>66.42</td>
</tr>
<tr>
<th rowspan="4">MagR</th>
<th>RTN</th>
<td>56320.00</td>
<td>68096.00</td>
<td>1696.00</td>
<td>35.55</td>
<td>35.15</td>
<td>36.15</td>
<td>10.56</td>
<td>9.06</td>
<td>7.50</td>
<td>55.31</td>
<td>59.89</td>
<td>64.93</td>
<td>10.12</td>
<td>8.62</td>
<td>7.28</td>
<td>56.66</td>
<td>61.02</td>
<td>66.33</td>
</tr>
<tr>
<th>GPTQ</th>
<td>43.75</td>
<td>40.00</td>
<td>13.81</td>
<td>40.42</td>
<td>42.54</td>
<td>51.59</td>
<td>9.94</td>
<td>8.38</td>
<td>7.34</td>
<td>58.19</td>
<td>62.56</td>
<td>65.39</td>
<td>9.62</td>
<td>8.25</td>
<td>7.16</td>
<td>58.04</td>
<td>61.89</td>
<td>66.12</td>
</tr>
<tr>
<th>Qronos</th>
<td>34.25</td>
<td>19.75</td>
<td>13.81</td>
<td>41.37</td>
<td>46.71</td>
<td>54.87</td>
<td>9.81</td>
<td>8.38</td>
<td>7.28</td>
<td>57.71</td>
<td>61.56</td>
<td>65.65</td>
<td>9.50</td>
<td>8.00</td>
<td>7.16</td>
<td>57.90</td>
<td>61.51</td>
<td>66.38</td>
</tr>
<tr>
<th>Learned Round</th>
<td>22.49</td>
<td>15.79</td>
<td></td>
<td><strong>46.83</strong></td>
<td>48.03</td>
<td></td>
<td>10.14</td>
<td>8.08</td>
<td></td>
<td>58.21</td>
<td>63.07</td>
<td></td>
<td>9.03</td>
<td>7.80</td>
<td></td>
<td>59.14</td>
<td>63.44</td>
<td></td>
</tr>
</tbody>
</table>


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
<td colspan="3"></td>
<th colspan="6">W4A4</th>
</tr>
<tr>
<td colspan="3"></td>
<th colspan="3">WikiText2 ↓</th>
<th colspan="3">0-shot ↑</th>
</tr>
<tr>
<th>Model</th>
<th>Stage 1</th>
<th>Stage 2</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
</tr>
<tr>
<th rowspan="21">Llama-3.2</th>
<th>BF16</th>
<td></td>
<td>8.9</td>
<td>7.2</td>
<td>5.9</td>
<td>56.2</td>
<td>63.6</td>
<td>69.1</td>
</tr>
<tr>
<th rowspan="4">None</th>
<th>RTN</th>
<td>6304.00</td>
<td>22016.00</td>
<td>52736.00</td>
<td>34.59</td>
<td>34.83</td>
<td>35.60</td>
</tr>
<tr>
<th>GPTQ</th>
<td>23424.00</td>
<td>14208.00</td>
<td>23424.00</td>
<td>34.38</td>
<td>35.48</td>
<td>34.32</td>
</tr>
<tr>
<th>Qronos</th>
<td>174.00</td>
<td>84.50</td>
<td>82.00</td>
<td>37.44</td>
<td>38.59</td>
<td>38.65</td>
</tr>
<tr>
<th>Learned Round</th>
<td>100.67</td>
<td>73.80</td>
<td>274.88</td>
<td>36.10</td>
<td>39.03</td>
<td>38.15</td>
</tr>
<tr>
<th rowspan="4">HIP</th>
<th>RTN</th>
<td>18.25</td>
<td>10.56</td>
<td>8.38</td>
<td>45.78</td>
<td>55.25</td>
<td>61.33</td>
</tr>
<tr>
<th>GPTQ</th>
<td>13.19</td>
<td>8.75</td>
<td>7.50</td>
<td>48.49</td>
<td>58.35</td>
<td>62.76</td>
</tr>
<tr>
<th>Qronos</th>
<td>13.19</td>
<td>9.19</td>
<td>7.62</td>
<td>48.40</td>
<td>58.24</td>
<td>62.85</td>
</tr>
<tr>
<th>Learned Round</th>
<td>12.32</td>
<td>8.78</td>
<td>7.23</td>
<td>50.57</td>
<td>59.09</td>
<td>63.70</td>
</tr>
<tr>
<th rowspan="4">MagR</th>
<th>RTN</th>
<td>5920.00</td>
<td>8096.00</td>
<td>24960.00</td>
<td>34.94</td>
<td>35.03</td>
<td>34.75</td>
</tr>
<tr>
<th>GPTQ</th>
<td>12544.00</td>
<td>17152.00</td>
<td>24960.00</td>
<td>35.74</td>
<td>35.91</td>
<td>35.44</td>
</tr>
<tr>
<th>Qronos</th>
<td>197.00</td>
<td>153.00</td>
<td>174.00</td>
<td>36.74</td>
<td>37.65</td>
<td>38.05</td>
</tr>
<tr>
<th>Learned Round</th>
<td>103.10</td>
<td>82.66</td>
<td></td>
<td>38.74</td>
<td>36.75</td>
<td></td>
</tr>
<tr>
<th rowspan="4">QuaRot</th>
<th>RTN</th>
<td>27.88</td>
<td>19.12</td>
<td>11.62</td>
<td>42.25</td>
<td>44.93</td>
<td>55.34</td>
</tr>
<tr>
<th>GPTQ</th>
<td>14.69</td>
<td>10.12</td>
<td>8.00</td>
<td>47.54</td>
<td>55.34</td>
<td>61.58</td>
</tr>
<tr>
<th>Qronos</th>
<td>13.81</td>
<td>9.31</td>
<td>7.75</td>
<td>48.77</td>
<td>57.18</td>
<td>62.80</td>
</tr>
<tr>
<th>Learned Round</th>
<td>13.65</td>
<td>9.88</td>
<td>7.86</td>
<td>49.26</td>
<td>55.22</td>
<td>44.84</td>
</tr>
<tr>
<th rowspan="4">SpinQuant</th>
<th>RTN</th>
<td>18.25</td>
<td>87.00</td>
<td>77.00</td>
<td>46.57</td>
<td>35.06</td>
<td>36.15</td>
</tr>
<tr>
<th>GPTQ</th>
<td>15.38</td>
<td>1240.00</td>
<td>392.00</td>
<td>47.50</td>
<td>34.58</td>
<td>34.67</td>
</tr>
<tr>
<th>Qronos</th>
<td>14.69</td>
<td>368.00</td>
<td>286.00</td>
<td>47.81</td>
<td>34.68</td>
<td>35.27</td>
</tr>
<tr>
<th>Learned Round</th>
<td>13.52</td>
<td>9.41</td>
<td>7.59</td>
<td>50.22</td>
<td>57.08</td>
<td>62.22</td>
</tr>
</tbody>
</table>

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
<th colspan="3"></th>
<th colspan="6">W4g32A</th>
</tr>
<tr>
<th colspan="3"></th>
<th colspan="3">WikiText2 ↓</th>
<th colspan="3">0-shot ↑</th>
</tr>
<tr>
<th>Model</th>
<th>Stage 1</th>
<th>Stage 2</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
<th>1B</th>
<th>3B</th>
<th>8B</th>
</tr>
<tr>
<th rowspan="21">Llama-3.2</th>
<th>BF16</th>
<td></td>
<td>8.9</td>
<td>7.2</td>
<td>5.9</td>
<td>56.2</td>
<td>63.6</td>
<td>69.1</td>
</tr>
<tr>
<th rowspan="4">None</th>
<th>RTN</th>
<td>14.44</td>
<td>9.19</td>
<td>7.75</td>
<td>50.15</td>
<td>57.39</td>
<td>63.45</td>
</tr>
<tr>
<th>GPTQ</th>
<td>12.38</td>
<td>8.62</td>
<td>7.16</td>
<td>51.80</td>
<td>56.95</td>
<td>64.68</td>
</tr>
<tr>
<th>Qronos</th>
<td>12.56</td>
<td>8.75</td>
<td>7.22</td>
<td>51.57</td>
<td>59.14</td>
<td>64.09</td>
</tr>
<tr>
<th>Learned Round</th>
<td>11.78</td>
<td>8.49</td>
<td>6.97</td>
<td>52.78</td>
<td>61.01</td>
<td>65.21</td>
</tr>
<tr>
<th rowspan="4">HIP</th>
<th>RTN</th>
<td>13.19</td>
<td>8.94</td>
<td>7.28</td>
<td>50.42</td>
<td>59.21</td>
<td>65.99</td>
</tr>
<tr>
<th>GPTQ</th>
<td>11.06</td>
<td>8.25</td>
<td>6.78</td>
<td>52.49</td>
<td>60.98</td>
<td>65.94</td>
</tr>
<tr>
<th>Qronos</th>
<td>11.62</td>
<td>8.50</td>
<td>7.06</td>
<td>51.58</td>
<td>59.72</td>
<td>65.54</td>
</tr>
<tr>
<th>Learned Round</th>
<td>11.01</td>
<td>8.38</td>
<td>6.70</td>
<td>53.05</td>
<td>61.11</td>
<td>65.64</td>
</tr>
<tr>
<th rowspan="4">MagR</th>
<th>RTN</th>
<td>18.88</td>
<td>12.00</td>
<td>8.94</td>
<td>46.03</td>
<td>48.84</td>
<td>57.59</td>
</tr>
<tr>
<th>GPTQ</th>
<td>14.44</td>
<td>9.94</td>
<td>7.88</td>
<td>49.36</td>
<td>53.12</td>
<td>62.19</td>
</tr>
<tr>
<th>Qronos</th>
<td>13.19</td>
<td>8.94</td>
<td>7.50</td>
<td>51.28</td>
<td>58.27</td>
<td>63.86</td>
</tr>
<tr>
<th>Learned Round</th>
<td>12.48</td>
<td>9.50</td>
<td></td>
<td>50.86</td>
<td>59.18</td>
<td></td>
</tr>
<tr>
<th rowspan="4">QuaRot</th>
<th>RTN</th>
<td>15.62</td>
<td>12.38</td>
<td>8.50</td>
<td>48.36</td>
<td>54.34</td>
<td>62.64</td>
</tr>
<tr>
<th>GPTQ</th>
<td>12.19</td>
<td>9.06</td>
<td>7.38</td>
<td>51.10</td>
<td>58.52</td>
<td>64.59</td>
</tr>
<tr>
<th>Qronos</th>
<td>11.81</td>
<td>8.62</td>
<td>7.00</td>
<td>51.71</td>
<td>59.06</td>
<td></td>
</tr>
<tr>
<th>Learned Round</th>
<td>11.69</td>
<td>8.40</td>
<td>6.86</td>
<td>52.35</td>
<td>60.26</td>
<td>41.28</td>
</tr>
<tr>
<th rowspan="4">SpinQuant</th>
<th>RTN</th>
<td>12.00</td>
<td>8.75</td>
<td>7.16</td>
<td>51.92</td>
<td>59.35</td>
<td>66.01</td>
</tr>
<tr>
<th>GPTQ</th>
<td>12.38</td>
<td>9.62</td>
<td>8.12</td>
<td>51.06</td>
<td>58.37</td>
<td>62.93</td>
</tr>
<tr>
<th>Qronos</th>
<td>11.62</td>
<td>8.62</td>
<td>7.22</td>
<td>51.50</td>
<td>59.26</td>
<td>64.49</td>
</tr>
<tr>
<th>Learned Round</th>
<td>11.71</td>
<td>8.51</td>
<td>6.93</td>
<td>52.52</td>
<td>59.79</td>
<td></td>
</tr>
</tbody>
</table>
