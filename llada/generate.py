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

import os

import numpy as np
import torch
import torch.nn.functional as F
from model.modeling_llada import LLaDAModelLM
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


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


@torch.no_grad()
def generate_with_drs_improved(model, prompt, steps=128, gen_length=128, block_length=128,
                               temperature=0., remasking='low_confidence', mask_id=126336,
                               threshold=0.8, t_base=8):
    """
    研究目的に沿った真のDRS実装

    目的:
    1. 難しいブロックの識別
    2. 動的な計算資源配分
    3. 早期終了によるNFE削減
    4. 品質保持

    修正点:
    1. 完成ブロックは絶対に変更しない
    2. 未完成ブロックのみに追加計算を集中
    3. 異常な「信頼度ベース精錬」を削除
    4. 真の早期終了を実現
    """
    # シーケンスをマスクで初期化
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    # Phase 1: 各ブロックにt_baseステップを配分
    block_confidences = []
    block_remaining_masks = []
    nfe = 0

    print(f"Phase 1: 初期パス - {t_base}ステップ x {num_blocks}ブロック")

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # 各ブロックでt_baseステップを実行
        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, t_base)

        final_confidence_scores = None

        for i in range(t_base):
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, current_block_end:] = 0

            x0, transfer_index, confidence_scores = get_transfer_index_with_confidence(
                logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i])

            x[transfer_index] = x0[transfer_index]

            # 最後のステップで信頼度スコアを保存
            if i == t_base - 1:
                final_confidence_scores = confidence_scores[0,
                                                            current_block_start:current_block_end]

        # 残りマスク数を記録
        remaining_masks = (
            x[:, current_block_start:current_block_end] == mask_id).sum().item()
        block_remaining_masks.append(remaining_masks)

        # 曖昧度スコア計算（未完成ブロックのみ）
        if remaining_masks > 0 and final_confidence_scores is not None:
            # マスクされていない有効なトークンの信頼度のみを考慮
            valid_mask = (
                x[0, current_block_start:current_block_end] != mask_id)
            if valid_mask.sum() > 0:
                valid_scores = final_confidence_scores[valid_mask]
                low_confidence_tokens = (valid_scores < threshold).float()
                ambiguity_score = low_confidence_tokens.mean().item()
            else:
                ambiguity_score = 1.0  # 全てマスクの場合は最大曖昧度
        else:
            ambiguity_score = 0.0  # 完成ブロックは曖昧度0

        block_confidences.append(ambiguity_score)

        # デバッグ情報
        if remaining_masks > 0:
            print(f"ブロック {num_block}: 残りマスク={remaining_masks}, "
                  f"曖昧度スコア={ambiguity_score:.3f} (未完成)")
        else:
            print(f"ブロック {num_block}: 完成済み, 曖昧度スコア=0.0")

    # Phase 2: 動的予算再配分
    t_used_base = t_base * num_blocks
    t_refine = max(0, steps - t_used_base)
    total_remaining_masks = sum(block_remaining_masks)

    print(f"\nPhase 2: 動的予算再配分")
    print(f"  使用済み予算: {t_used_base}")
    print(f"  残り予算: {t_refine}")
    print(f"  ブロック曖昧度スコア: {[f'{s:.3f}' for s in block_confidences]}")
    print(f"  ブロック残りマスク数: {block_remaining_masks}")
    print(f"  総残りマスク数: {total_remaining_masks}")

    # 早期終了の条件チェック
    if total_remaining_masks == 0:
        print(f"  → 全ブロック完成。早期終了")
        print(f"  → NFE効率: {nfe}/{steps} = {(nfe/steps*100):.1f}%")
        return x, nfe, block_confidences

    if t_refine <= 0:
        print(f"  → 予算不足。現在の状態で終了")
        print(f"  → NFE効率: {nfe}/{steps} = {(nfe/steps*100):.1f}%")
        return x, nfe, block_confidences

    # 未完成ブロックのみを対象とした曖昧度に基づく予算配分
    incomplete_blocks = [i for i, masks in enumerate(
        block_remaining_masks) if masks > 0]
    if not incomplete_blocks:
        print(f"  → 処理すべき未完成ブロックなし")
        return x, nfe, block_confidences

    incomplete_ambiguities = [block_confidences[i] for i in incomplete_blocks]
    total_ambiguity = sum(incomplete_ambiguities)

    if total_ambiguity == 0:
        # 曖昧度がない場合は均等配分
        steps_per_incomplete = t_refine // len(incomplete_blocks)
        additional_steps = [steps_per_incomplete if i in incomplete_blocks else 0
                            for i in range(num_blocks)]
    else:
        # 曖昧度に比例した配分
        additional_steps = [0] * num_blocks
        allocated_total = 0

        for idx, block_idx in enumerate(incomplete_blocks):
            if idx == len(incomplete_blocks) - 1:
                # 最後のブロックに残り予算を配分
                steps = t_refine - allocated_total
            else:
                ambiguity = incomplete_ambiguities[idx]
                steps = int((ambiguity / total_ambiguity) * t_refine)
                allocated_total += steps

            additional_steps[block_idx] = max(0, steps)

    print(f"  → 追加ステップ配分: {additional_steps}")

    # Phase 3: 未完成ブロックの標的精錬
    if any(steps > 0 for steps in additional_steps):
        print(f"\nPhase 3: 未完成ブロックの標的精錬")

        for num_block, steps_to_add in enumerate(additional_steps):
            if steps_to_add == 0:
                continue

            current_block_start = prompt.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            # このブロックに精錬すべきマスクが残っているか確認
            remaining_in_block = (
                x[:, current_block_start:current_block_end] == mask_id).sum().item()
            if remaining_in_block == 0:
                print(f"  → ブロック {num_block} は既に完成済み、スキップ")
                continue

            print(f"  → ブロック {num_block} を {steps_to_add} ステップで精錬...")
            print(f"    精錬前残りマスク: {remaining_in_block}")

            for step in range(steps_to_add):
                # このブロックが精錬中に完成したらループを抜ける
                current_masks = (
                    x[:, current_block_start:current_block_end] == mask_id).sum().item()
                if current_masks == 0:
                    print(f"    ブロック {num_block} の精錬完了 (ステップ {step+1}で完成)")
                    break

                nfe += 1
                mask_index = (x == mask_id)
                logits = model(x).logits
                mask_index[:, current_block_end:] = 0

                # 現在のブロック内のマスクのみを対象とした転送数計算
                block_mask_index = (
                    x[:, current_block_start:current_block_end] == mask_id)
                remaining_steps = steps_to_add - step
                block_transfer_tokens = get_num_transfer_tokens(
                    block_mask_index, remaining_steps)

                x0, transfer_index, _ = get_transfer_index_with_confidence(
                    logits, temperature, remasking, mask_index, x, block_transfer_tokens[:, 0])

                x[transfer_index] = x0[transfer_index]

            # 精錬後の状態確認
            final_masks = (
                x[:, current_block_start:current_block_end] == mask_id).sum().item()
            print(f"    精錬後残りマスク: {final_masks}")

    # 最終結果
    final_total_masks = (x[:, prompt.shape[1]:] == mask_id).sum().item()
    completion_rate = ((gen_length - final_total_masks) / gen_length) * 100

    print(f"\n真のDRS生成完了:")
    print(f"  総NFE: {nfe}/{steps} ({(nfe/steps*100):.1f}%)")
    print(
        f"  完成率: {completion_rate:.1f}% ({gen_length - final_total_masks}/{gen_length})")
    print(f"  早期終了による削減: {steps - nfe} NFE")
    print(f"  曖昧度分散: {np.var(block_confidences):.3f}")

    return x, nfe, block_confidences


# 後方互換性のため、既存の関数名をエイリアスとして残す
def generate_with_drs_fixed(model, prompt, steps=128, gen_length=128, block_length=128,
                            temperature=0., remasking='low_confidence', mask_id=126336,
                            threshold=0.8, t_base=8):
    """後方互換性のためのエイリアス - 新しい研究版を呼び出し"""
    return generate_with_drs_improved(model, prompt, steps, gen_length, block_length,
                                      temperature, remasking, mask_id, threshold, t_base)


def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128,
                                   block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(
        out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
