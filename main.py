import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers, prune_gradient, prune_gblm
from lib.eval import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    print("printing gpu allocation for all the layers")
    print(model.hf_device_map)
    model.seqlen = 2048

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--gradient_path', default=None,type=str, help='gradient path')
    parser.add_argument('--grad_norm', type=str, default="none", choices=["none", "accumulation_norm", "2-norm-sample-dim"])
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--seq_length', type=int, default=2048, help='Sequence length of the input.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--layer_no', type=int, default=-1, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt","gradient", "gblm"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--grad_exponent', action='store_true', help='Use gradient of exponent')
    parser.add_argument('--gradient_inv', action='store_true', help='Use inverse of gradient')
    args = parser.parse_args()
    print(f"Working on model: {args.model}")
    print(f"working on method {args.prune_method}, grad norm {args.grad_norm}, gradient path {args.gradient_path}, inverse enabled {args.gradient_inv}, sparsity type {args.sparsity_type}, seq lenght {args.seq_length}")

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model or "70b" in args.model: 
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    idx = args.layer_no
    print(f"pruning for sparsity_ratio {args.sparsity_ratio} by method {args.prune_method}")
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, layer_no=idx)
        elif args.prune_method == "gblm":
            prune_gblm(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, layer_no=idx)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, layer_no=idx)
        elif args.prune_method == "gradient":
            prune_gradient(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, layer_no=idx)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, layer_no=idx)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model, args)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl = eval_ppl(model, tokenizer, device)
    print(f"ppl on wikitext {ppl}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, "log.txt")
    with open(save_filepath, "w") as f:
        print("actual_sparsity\tppl", file=f, flush=True)
        print(f"{sparsity_ratio:.4f}\t{ppl:.4f}", file=f, flush=True)
    
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
    print("*"*30)

if __name__ == '__main__':
    main()