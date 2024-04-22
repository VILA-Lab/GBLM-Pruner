#  GBLM-Pruner: Gradient Based Language Model Pruner

*[Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models.](https://arxiv.org/abs/2311.04902)*

[Rocktim Jyoti Das*](https://scholar.google.com/citations?user=kDsyWlMAAAAJ&hl=en), [Mingjie Sun*](https://eric-mingjie.github.io/), [Liqun Ma](https://scholar.google.com/citations?user=zVXXXGIAAAAJ&hl=zh-CN), [Zhiqiang Shen*](http://zhiqiangshen.com/)

Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi.

Carnegie Mellon University, Pittsburgh.

## Contents
- [Introduction](#introduction)
- [Install](#install)
- [GBLM-Pruned Weights](#gblm-pruned-weights)
- [Model Zoo](https://github.com/RocktimJyotiDas/GBLM-Pruner/blob/main/MODEL_ZOO.md)
- [Usage](#usage)
- [Zero-Shot Harness Evaluation](#zero-shot-harness-evaluation)
- [Acknowledgement](#Acknowledgement)
- [Issues](#issues)
- [License](#license)
- [Citation](#citation)

## Introduction

Gradient information has been overlooked by prior methods for neural model pruning. Even the original Optimal Brain Surgeon work ignored first-order term in the derivation of OBS framework for model pruning. It was done under the assumption that gradients at the minimum vanish and cease to offer any valuable information. In this work, we revisited and refined the OBS framework by incorporation consideration of first-order-term. Based on our analysis, we propose our gradient based pruning metric.

## Install

The installation instructions are provided [here](https://github.com/RocktimJyotiDas/GBLM-Pruner/blob/main/install.md).

## GBLM-Pruned Weights
Please check out our [Model Zoo](https://github.com/RocktimJyotiDas/GBLM-Pruner/blob/main/MODEL_ZOO.md) for all public GBLM-Pruner compressed model checkpoints, and the instructions of how to use the weights.

## Usage
Our method require computation of gradient magnitude for calculation of pruning metric. The gradient for a model can be computed as follows:
```sh
bash run_grad_compute.sh
```
Overview of the arguments in the bash file:  
- `--model`: The identifier or the path for the LLaMA model.
- `--llama_version`: Version of Llama model using (for LLaMA-1 use 1 and for LLaMA-2 use 2)
- `--nsamples`: No of calibration samples.

After computation of the model gradient, the pruned model can be obtained using the following command. 
```sh
bash run_gblm_prune.sh
```
Overview of the arguments in the bash file:  
- `--model`: The identifier or the path for the LLaMA model.
- `--gradient_path`: Path to the pre-computed gradient 
- `--prune_method`: Pruning method to be used.
- `--nsamples`: No of calibration samples.
- `--seed`: Random seed.
- `--sparsity_ratio`: Percentage of the weights to be pruned.
- `--sparsity_type`: Specify the sparsity type.
- `--save`: Path to store results.

## Zero-Shot Harness Evaluation

We use the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) implementation for the zero-shot evaluation on Harness. We used the same instructions provided [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/README.md) for producing our results. We used the following command for reproducing our results.
```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/path/to/model \
    --tasks task_name \
    --device cuda:0 \
    --no_cache
```


## Acknowledgement

This codebase is built upon [SparseGPT](https://github.com/IST-DASLab/sparsegpt) and [Wanda](https://github.com/locuslab/wanda).

## Issues
Please don't hesitate to contact us if you encounter any code-related issues or wish to discuss the paper. You can reach out to us via the [GitHub issues](https://github.com/RocktimJyotiDas/GBLM-Pruner/issues)  or through email at rocktimjyotidas@gmail.com.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation

If you found this work useful, please consider citing:

```
@misc{das2023size,
      title={Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models}, 
      author={Rocktim Jyoti Das and Liqun Ma and Zhiqiang Shen},
      year={2023},
      eprint={2311.04902},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
