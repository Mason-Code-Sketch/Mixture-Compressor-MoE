<p align="center" width="10%">
<img src="imgs/logo.png" style="width: 30%" align=center> 
</p>

# Mixture Compressor for Mixture-of-Experts LLMs Gains More

[Wei Huang](https://aaron-weihuang.com/), [Yue Liao](https://scholar.google.com/citations?user=mIt-3fEAAAAJ&hl=en), [Jianhui Liu](https://scholar.google.com/citations?user=n1JW-jYAAAAJ&hl=en), [Ruifei He](https://scholar.google.com/citations?user=P7IL0hkAAAAJ), [Haoru Tan](), [Shiming Zhang*](), [Hongsheng Li](https://scholar.google.com/citations?user=BN2Ze-QAAAAJ&hl=zh-CN), [Si Liu*](https://scholar.google.com/citations?user=-QtVtNEAAAAJ&hl=en) and [Xiaojuan Qi*](https://scholar.google.com/citations?user=bGn0uacAAAAJ&hl=en) (* corresponding author)

[![arXiv](https://img.shields.io/badge/MCMoE-2410.06270-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06270)



![WX20241009-191322@2x](imgs/WX20241009-191322@2x.png)

Extreme compression of Mixture-of-Experts Large Language Models. The current release supports:

- MC-MoE for accurate weight-only quantization (Weight=**1.5～2.5bit**).
- MC-MoE for efficient online dynamic pruning (additional compression ratio **> 10%**)
- Support pre-trained MoE-LLM (currently only support **Mixtral 8 $\times$ 7b** and **Mixtral 8 $\times$ 22b**)
- Real deployment and dequantization process is built on `HQQ` and `GPTQ` for vanilla group-wise quantization. MC-MoE's static quantization can also be transfered to other type of technique, such as Vector Quantization (codebook).

## Install
```sh
conda create -n mcmoe python=3.10 -y
conda activate mcmoe
git clone https://github.com/Aaronhuang-778/MC-MoE
cd MC-MoE
pip install --upgrade pip 
pip install -r requirements.txt
```

For real quantization and deployment of compressed model, we utilize [HQQ](https://github.com/mobiusml/hqq) to dequant the MoE LLMs with 1/2/3-bits. We changed the storage and dequantization process of 1-bit weights in HQQ. 

Please make sure you have a Pytorch 2 version that matches your CUDA version: https://pytorch.org/.

## Experts Precision Zoo
We provide the solved bit-width for each expert of **Mixtral 8 $\times$ 7b** in `./experts_mixture_bit_selection/`. You can directly use our provided results or generate them by your self.

## Usage

**Quickly get the compressed model `./scripts/quant.sh`**. We have already provided all the needed middle results of **Mixtral 8 $\times$ 7b** in this code.

```sh
# Replace the path with yours, for example:
Model_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1"
Saving_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1-2.5b"
Precision_Path="./experts_mixture_bit_selection/experts_mixture_bitwidth_combination_20bit.pkl"
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}

```

Efficient Inference with **Pre-Loading MixedPrecision Quantization** and **Online Dynamic Pruning** : This example shows the demo of 2.5bit  Mixtral-8x7B model, the total static GPU memory consumption is around **16GB**, and the running memory is around **19GB** .

```python
import os
import torch
from transformers import AutoTokenizer
from inference import load_quantized_model
from expert_weight import  replace_with_dynamic_rank
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.is_available()


kwargs = {"device_map": 'auto',
          "torch_dtype": "torch.float16"}
######## Input your save_dir of quantized model########
save_dir = "/mnt/models/mistralai/Mixtral-8x7B-v0.1-2.5bit"
model = load_quantized_model(save_dir, kwargs)

######### Choose if you want to use dynamic pruning or not ##########
args = None
model = replace_with_dynamic_rank(model, args, block_range=10)
######### Choose if you want to use dynamic pruning or not ##########

tokenizer = AutoTokenizer.from_pretrained(save_dir)
prompt = "You are a writer. Please write a short story about two llamas in a forest"
prompt_template=f'''{prompt}
'''

inputs = tokenizer(prompt_template, return_tensors="pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
inputs.input_ids = inputs.input_ids.to(device)

inputs.attention_mask = inputs.attention_mask.to(device)
# Generate
outputs = model.generate(inputs.input_ids, 
                        max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        )

print(tokenizer.decode(outputs[0]))
```



**Detailed process of MC-MOE**. 

1. First, we need to generate the expert factors for bit-width allocation in each MoE block:

   Download first part of the C4 training data `c4-train.00000-of-01024.json` from [allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz). Please save it at `./data`  and organize the datasets as follows: 
```
./data
|-- build.py
|-- c4-train.00000-of-01024.json
|-- dataset.py
|-- math_calib_construction.py
`-- math_pretrain_style.json
```
Run the  `./scripts/factors.sh` to generate the `activated frequencies`, `activated weights`   and `quantization loss` of each experts.

```sh
# Please run this script in ./scripts/factors.sh

# Your local model path
Model_Path=""
python awareness.py ${Model_Path} --calibration c4

# for example:
Model_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1"
python awareness.py ${Model_Path} --calibration c4

```
You can also change the `--calibration` data to `math` to make the calibration for specific knowledge and task. We have already provided the factors file from `c4` data set, please check:

```
|-- experts_act_frequency.pkl
|-- experts_act_weight.pkl
|-- experts_quant_loss.pkl
```

2. Second, we use the factors to solve the bit-width allocation problem through `precision_solver.py` (please generate the previous factor files before run the bit-width allocation code):

```sh
python precision_solver.py
```

This solver will save the optimal configurations of experts in each MoE block:

```sh
./experts_mixture_bit_selection
|-- experts_mixture_bitwidth_combination_12bit.pkl
|-- experts_mixture_bitwidth_combination_13bit.pkl
|-- experts_mixture_bitwidth_combination_14bit.pkl
|-- experts_mixture_bitwidth_combination_15bit.pkl
|-- experts_mixture_bitwidth_combination_16bit.pkl
|-- experts_mixture_bitwidth_combination_17bit.pkl
|-- experts_mixture_bitwidth_combination_18bit.pkl
|-- experts_mixture_bitwidth_combination_19bit.pkl
|-- experts_mixture_bitwidth_combination_20bit.pkl

# 12 bit means the total bit-width of 8 experts in one MoE block, the average bit-width of experts is 12/8=1.5bit.
```

3. Third, we can run the quantization code to compress the static parameters of MoE LLMs:

Run the  `./scripts/quant.sh` to quantize the MoE LLMs and test the perplexity results.

```sh
# Your local model path
Model_Path=""
# Expected experts precisions file
Precision_Path=""
##### fake quantization to test the performance of MC-MoE #####
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path}

# for example:
Model_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1"
Precision_Path="./experts_mixture_bit_selection/experts_mixture_bitwidth_combination_16bit.pkl"
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path}

```

add `--pach` and `--save` to save the real quantized model :

```sh
# Your local model path
Model_Path=""
# Your expected saving path
Saving_Path=""
# Expected experts precisions file
Precision_Path=""

##### real quantization and model pack for compact storage #####
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}

# for example:
Model_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1"
Saving_Path="/mnt/models/mistralai/Mixtral-8x7B-v0.1-2.05bit"
Precision_Path="./experts_mixture_bit_selection/experts_mixture_bitwidth_combination_16bit.pkl"
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}

```

4. Finally, you can load the quantized MoE LLMs and enable dynamic pruning inference with our provided inference demo:

```sh
python inference_demo.py
```



where you can also choose whether to enable dynamic pruning to improve inference efficiency  (line `17~18`)：

```python
from expert_weight import  replace_with_dynamic_rank
args = None
model = replace_with_dynamic_rank(model, args, block_range=10)
```
Detailed inference demo:

```python
import os
import torch
from transformers import AutoTokenizer
from inference import load_quantized_model
from expert_weight import  replace_with_dynamic_rank
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.is_available()


kwargs = {"device_map": 'auto',
          "torch_dtype": "torch.float16"}
######## Input your save_dir of quantized model########
save_dir = "/mnt/models/mistralai/Mixtral-8x7B-v0.1-2.5bit"
model = load_quantized_model(save_dir, kwargs)

######### Choose if you want to use dynamic pruning or not ##########
args = None
model = replace_with_dynamic_rank(model, args, block_range=10)
######### Choose if you want to use dynamic pruning or not ##########

tokenizer = AutoTokenizer.from_pretrained(save_dir)
prompt = "You are a writer. Please write a short story about two llamas in a forest"
prompt_template=f'''{prompt}
'''

inputs = tokenizer(prompt_template, return_tensors="pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
inputs.input_ids = inputs.input_ids.to(device)

inputs.attention_mask = inputs.attention_mask.to(device)
# Generate
outputs = model.generate(inputs.input_ids, 
                        max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        )

print(tokenizer.decode(outputs[0]))
```



## Evaluation

![m2](imgs/m2.png)

We use the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/2a47159caff00135b026f724ace2a2011f3c7621) (commit 2a47159) framework to evaluate the performance of fake quantized MoE LLMs. The command we use for LM Harness evaluation is as follows:

```sh
# accelerate launch \
#     --num_processes=1 \
#     --ipex \
#     -m lm_eval --model hf \
#     --model_args pretrained=model_path,dtype=float16,parallelize=True \
#     --tasks piqa,boolq,arc_challenge,arc_easy,hellaswag,winogrande,mmlu,mathqa \
#     --batch_size 32 \
```



## Related Project

[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://github.com/OpenGVLab/OmniQuant)

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models](https://github.com/Aaronhuang-778/SliM-LLM)

[Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models](https://github.com/Lucky-Lance/Expert_Sparsity)

[Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark](https://github.com/UNITES-Lab/moe-quantization)

[Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq)



## Native PMQ adapter on host 82

The `pmq-82` branch extends the original static PMQ method to DeepSeek-V2-Lite
and Qwen1.5-MoE-A2.7B. It keeps the original significance-weighted expert objective,
1/2/3-bit candidates, and per-layer middle/high-bit constraints. It does not add
router fine-tuning or online dynamic pruning.

Run commands from the `PMQ` worktree using the environment in the main checkout:

```sh
/root/moe-ptq-experiment/third-party/Mixture-Compressor-MoE/.venv/bin/python pmq_runner.py --help
```

Phases are explicit: `factors`, `allocate`, `quantize`, `ppl`, and `zeroshot`.
`evaluate` retains the combined quantization-plus-PPL entry point. A phase starts
only when invoked. The default output root remains the model's YAML `output_dir`;
`--out-dir` and the original-style `--saving-path` select other output directories.

The original artifact forms are preserved:

* `experts_act_frequency.pkl`, `experts_act_weight.pkl`, `experts_quant_loss.pkl`;
* `experts_mixture_bit_selection/experts_mixture_bitwidth_combination_avg<bit>bit.pkl`,
  containing the original layer -> expert -> integer-bit mapping;
* a model directory containing `qmodel.pt`, `config.json`, and tokenizer files.

The `avg` allocation filename identifies the requested average for architectures
with different expert counts. JSON sidecars record checksums, source/configuration
identity, actual parameter-weighted expert precision, and completion. They do not
replace the original PKLs or packed model. Existing outputs are not overwritten.
`inference.load_quantized_model` also recognizes the adapted checkpoint format.

DeepSeek is loaded through native Transformers experts, with the original YaRN
attention scale retained when needed. Both list-based and fused expert layouts are
supported. GPTQ scales/zeros are solved in the original FP16 arithmetic and stored
in FP32 metadata without another parameter search or quantization.

The DeepSeek-V2-Lite configuration uses `budget_policy: balanced_exact`. It
solves floor/ceiling layer budgets exactly and assigns the required extra units
to layers with the largest PMQ-objective improvement, breaking equal-gain ties
by ascending layer index. With 26 MoE layers and 66 parameter-equivalent expert
units per layer, the 1.25-bit setting assigns total budgets of 82 to 13 layers
and 83 to 13 layers, including the fixed shared-expert cost. The 1.75/2.25-bit
settings similarly use 115/116 and 148/149. All six settings match their requested
expert average bit exactly. Allocation metadata records actual average bit,
per-layer budgets, policy, and allocation time.

This rule is an explicit architecture budget extension. `strict` retains integer
per-layer targets, while `floor` keeps per-layer upper bounds and records the lower
actual precision. Other configurations retain their existing policy.

`pmq_queue.py --configs ... --dry-run` prints the planned native commands without
running them. An explicitly invoked queue reuses verified artifacts, runs all PPL
settings before seven-task evaluations for each model, and stops on errors.
Partial files are retained for diagnosis. The queue and individual allocation
phase both use the DeepSeek configuration policy without a command-line override.

Seven-task evaluation uses `lm_eval==0.4.13`, the seven comparison tasks, zero-shot,
no limit/chat template, and explicit `add_bos_token=False`; MMLU is counted once.

Each phase writes its console/progress log and resource samples under `logs/`.
RAM RSS, available system memory, system swap, physical-GPU usage and process CUDA
allocation/reservation are distinguished. Sampling maxima are labelled as sampled
values. The monitor observes resources without changing limits or killing jobs.


### Resident grouped evaluation

Saved PMQ checkpoints are quantized once. Independent `ppl` and `zeroshot`
phases reload the same `qmodel.pt`; FP16/BF16 evaluation decodes projections once
and uses `torch._grouped_mm` for routed experts on DeepSeek-V2-Lite, Qwen1.5-MoE,
Qwen3-MoE and Mixtral. Native router, shared-expert and accumulation behavior is
preserved. These evaluation-only buffers are not saved back to the checkpoint.
Seven-task model loading explicitly uses SDPA for Qwen1.5-MoE, Qwen3-MoE and
Mixtral, and eager attention for DeepSeek-V2-Lite. Scoring, quantization and PPL
retain their existing eager configuration. `zeroshot.json` records the actual
`attn_implementation`; saved checkpoint settings and weights are preserved.

```sh
python pmq_runner.py --config configs/qwen3-30b-a3b-base.yaml --phase quantize --average-bit 2.5
python pmq_runner.py --config configs/qwen3-30b-a3b-base.yaml --phase ppl --average-bit 2.5
python pmq_runner.py --config configs/qwen3-30b-a3b-base.yaml --phase zeroshot --average-bit 2.5
```

The `quantize` phase uses an existing allocation. Both evaluation JSONs record
`evaluation_backend` and the source checkpoint SHA256. FP32 CPU validation models
retain the original PyTorch backend. Existing checkpoint/result files are never
overwritten.

### Component resource records

`RunMonitor` retains each phase's `.log`, `_resources.jsonl` samples and
`_resources.json` summary in the existing log directory. Native phase commands
now also record `comp_stats`, `solve_lp`, `quant_gptq`, model/data loading,
checkpoint I/O, per-dataset PPL and seven-task components. Core elapsed seconds
and percentage shares exclude loading, saving and evaluation. Shared factor
collection is counted once when assembling a multi-budget resource table.

Component times synchronize CUDA before starting and stopping. Peak reserved
memory comes from the process's PyTorch allocator; per-GPU peak sums are upper
bounds and exclude driver/non-PyTorch allocations. The existing sampled
`nvidia-smi` whole-device memory remains a separate field and may include other
processes. Process-tree RSS uses the configured sampling interval (default five
seconds), may double-count shared pages, and can miss short-lived peaks.

Attempts carry unique `run_id` values. Reruns archive the previous log, samples
and summary, then start fresh. Failed components and sampling errors are
explicitly recorded. A hard-killed attempt can leave a `running` summary with
its active component; it must not be treated as completed. Resource summaries
are written atomically at component boundaries and command exit. Historical
runs without allocator records cannot be assigned measured peaks retroactively.
