# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
from tqdm import tqdm

# ADAPTIVE SCHEDULING INTEGRATION
# try:
#     from generate_adaptive import generate_with_adaptive_scheduling
#     from adaptive_scheduler import AdaptiveInferenceScheduler
#     from cache_manager import TieredCacheManager
#     ADAPTIVE_SCHEDULING_AVAILABLE = True
# except ImportError:
#     ADAPTIVE_SCHEDULING_AVAILABLE = False

# GTS CONTROLLED SAMPLING INTEGRATION
try:
    from sampler.gts_controlled_sampler import generate_with_gts_controlled_sampling
    GTS_SAMPLING_AVAILABLE = True
except ImportError:
    GTS_SAMPLING_AVAILABLE = False


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(
        0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block *
                            block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] +
                       (num_block + 1) * block_length:] = 0
            x0, transfer_index = get_transfer_index(
                logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe


@torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                               remasking='low_confidence', mask_id=126336, threshold=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(
            output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i]
                                           [j][:, :, :current_block_start],)

        past_key_values = new_past_key_values
        nfe += 1

        i = 1
        while True:
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:],
                           past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(
                logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index,
                                                    x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1

    return x, nfe


@torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                             remasking='low_confidence', mask_id=126336, threshold=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(
            output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            nfe += 1
            mask_index = (
                x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values,
                           use_cache=True, replace_position=replace_position).logits

            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index,
                                                    x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1

    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_with_confidence(logits, temperature, remasking, mask_index, x, num_transfer_tokens):
    """
    Modified version of get_transfer_index that also returns confidence scores.

    Returns:
        (x0, transfer_index, confidence_scores)
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True

    return x0, transfer_index, x0_p


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description='LLaDA生成with multiple samplers')
    parser.add_argument('--sampler', type=str, default='dual_cache',
                        choices=['dual_cache', 'gts', 'compare'],
                        help='使用するサンプラー (default: dual_cache)')
    parser.add_argument('--prompt', type=str,
                        default="Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
                        help='生成に使用するプロンプト')
    parser.add_argument('--gen_length', type=int, default=128,
                        help='生成する長さ (default: 128)')
    parser.add_argument('--gts_threshold', type=float, default=0.8,
                        help='GTS制御サンプリングの閾値 (default: 0.8)')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='GTS制御サンプリングの最大反復回数 (default: 3)')
    parser.add_argument('--verbose', action='store_true',
                        help='詳細な出力を表示')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用するデバイス (default: cuda)')

    args = parser.parse_args()

    device = args.device

    # モデルとトークナイザーのロード
    print(f"🤖 モデルロード中...")
    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # プロンプト処理
    prompt = args.prompt
    m = [{"role": "user", "content": prompt}, ]
    formatted_prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    print(f"📝 プロンプト: {prompt}")
    print(f"🎯 サンプラー: {args.sampler}")
    print(f"📏 生成長: {args.gen_length}")
    print("=" * 50)

    # サンプラー選択と実行
    if args.sampler == 'dual_cache':
        print("🚀 Dual Cache サンプリング実行中...")
        start_time = time.time()
        out, nfe = generate_with_dual_cache(
            model, input_ids, steps=128, gen_length=args.gen_length,
            block_length=32, temperature=0., remasking='low_confidence')
        generation_time = time.time() - start_time

        generated_text = tokenizer.batch_decode(
            out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        print(f"✅ 生成完了 (NFE: {nfe}, 時間: {generation_time:.2f}s)")
        print(f"📄 生成結果:\n{generated_text}")

    elif args.sampler == 'gts':
        if not GTS_SAMPLING_AVAILABLE:
            print("❌ GTS制御サンプリングが利用できません")
            return

        print("🎯 GTS制御サンプリング実行中...")
        start_time = time.time()
        out, metrics = generate_with_gts_controlled_sampling(
            model, input_ids,
            gen_length=args.gen_length,
            gts_threshold=args.gts_threshold,
            max_iterations=args.max_iterations,
            verbose=args.verbose)
        generation_time = time.time() - start_time

        generated_text = tokenizer.batch_decode(
            out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        print(
            f"✅ 生成完了 (NFE: {metrics['nfe']}, 反復: {metrics['iterations']}, 時間: {generation_time:.2f}s)")
        print(
            f"📊 最終GTS: {metrics['gts_scores'][-1] if metrics['gts_scores'] else 'N/A':.4f}")
        print(f"📄 生成結果:\n{generated_text}")

    elif args.sampler == 'compare':
        print("📊 全手法比較実行中...")

        # 比較実行
        results = compare_generation_methods(
            model, input_ids, gen_length=args.gen_length, verbose=args.verbose)

        # GTS制御サンプリングも追加
        if GTS_SAMPLING_AVAILABLE:
            print("🎯 GTS制御サンプリングも比較に追加...")
            start_time = time.time()
            gts_out, gts_metrics = generate_with_gts_controlled_sampling(
                model, input_ids,
                gen_length=args.gen_length,
                gts_threshold=args.gts_threshold,
                max_iterations=args.max_iterations,
                verbose=False)
            gts_time = time.time() - start_time

            results['gts'] = {
                'output': gts_out,
                'nfe': gts_metrics['nfe'],
                'time': gts_time,
                'method': 'GTS Controlled',
                'iterations': gts_metrics['iterations'],
                'final_gts': gts_metrics['gts_scores'][-1] if gts_metrics['gts_scores'] else 0.0
            }

        # 結果表示
        print("\n" + "=" * 60)
        print("📈 最終比較結果")
        print("=" * 60)

        for method_name, result in results.items():
            if method_name != 'comparison':
                method = result['method']
                nfe = result['nfe']
                time_taken = result['time']
                print(f"{method:20s}: NFE={nfe:3d}, 時間={time_taken:6.2f}s")

                if method_name == 'gts' and 'final_gts' in result:
                    print(
                        f"{'':20s}  反復={result['iterations']}, 最終GTS={result['final_gts']:.4f}")

        # 生成結果の表示
        print(f"\n📄 生成結果 (dual_cacheで生成):")
        generated_text = tokenizer.batch_decode(
            results['dual_cache']['output'][:, input_ids.shape[1]:],
            skip_special_tokens=True)[0]
        print(generated_text)


@torch.no_grad()
def compare_generation_methods(model, prompt, gen_length=128, verbose=True):
    """
    各生成手法の比較を実行 (Dual Cache + GTS)
    """
    import time

    results = {}

    if verbose:
        print("🔍 生成手法比較開始")
        print("=" * 50)

    # 1. Dual Cache 生成
    if verbose:
        print("📊 generate_with_dual_cache 実行中...")

    start_time = time.time()
    dual_cache_output, dual_cache_nfe = generate_with_dual_cache(
        model, prompt, steps=128, gen_length=gen_length,
        block_length=32, temperature=0., remasking='low_confidence'
    )
    dual_cache_time = time.time() - start_time

    results['dual_cache'] = {
        'output': dual_cache_output,
        'nfe': dual_cache_nfe,
        'time': dual_cache_time,
        'method': 'Dual Cache (Static)'
    }

    return results


if __name__ == '__main__':
    main()
