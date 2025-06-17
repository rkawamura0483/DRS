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


def calculate_block_ambiguity(confidence_scores, threshold, mask_id):
    """
    Calculate ambiguity score for a block based on confidence threshold.

    Args:
        confidence_scores: Tensor of confidence scores for block tokens
        threshold: Confidence threshold τ
        mask_id: Mask token ID to exclude from calculation

    Returns:
        Ambiguity score (float): fraction of tokens below threshold
    """
    # Only consider non-mask tokens (valid predictions)
    valid_scores = confidence_scores[confidence_scores != -np.inf]
    if len(valid_scores) == 0:
        return 0.0

    # Calculate fraction below threshold
    low_confidence_tokens = (valid_scores < threshold).float()
    ambiguity_score = low_confidence_tokens.mean().item()

    return ambiguity_score


def allocate_refinement_budget(block_ambiguities, total_refinement_budget):
    """
    Allocate refinement budget proportionally to block ambiguity scores.

    Args:
        block_ambiguities: List of ambiguity scores per block
        total_refinement_budget: Total additional steps to distribute

    Returns:
        List of additional steps per block
    """
    total_ambiguity = sum(block_ambiguities)

    if total_ambiguity == 0:
        # No ambiguity, distribute equally
        steps_per_block = total_refinement_budget // len(block_ambiguities)
        return [steps_per_block] * len(block_ambiguities)

    # Proportional allocation
    additional_steps = []
    allocated_total = 0

    for i, ambiguity in enumerate(block_ambiguities):
        if i == len(block_ambiguities) - 1:
            # Last block gets remaining budget
            steps = total_refinement_budget - allocated_total
        else:
            # Proportional allocation
            steps = int((ambiguity / total_ambiguity)
                        * total_refinement_budget)
            allocated_total += steps

        additional_steps.append(max(0, steps))

    return additional_steps


@torch.no_grad()
def generate_with_drs_fixed(model, prompt, steps=128, gen_length=128, block_length=128,
                            temperature=0., remasking='low_confidence', mask_id=126336,
                            threshold=0.8, t_base=8):
    """
    修正版Dynamic Refinement Scheduling generation.

    主な修正点:
    1. 再マスキング戦略を削除（安定性向上）
    2. 真の早期終了を実装（NFE削減）
    3. 動的予算配分の改善
    """
    # シーケンスをマスクで初期化
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    # Phase 1: 初期粗いパス
    block_confidences = []
    nfe = 0

    print(f"Phase 1: 初期粗いパス開始 - {t_base}ステップ x {num_blocks}ブロック")

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # T_baseステップを実行
        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, t_base)

        block_confidence_scores = None

        for i in range(t_base):
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, current_block_end:] = 0

            x0, transfer_index, confidence_scores = get_transfer_index_with_confidence(
                logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i])

            x[transfer_index] = x0[transfer_index]

            # 最後のステップでブロック信頼度スコアを保存
            if i == t_base - 1:
                block_confidence_scores = confidence_scores[0,
                                                            current_block_start:current_block_end]

        # ブロック曖昧度スコア計算
        if block_confidence_scores is not None:
            ambiguity_score = calculate_block_ambiguity(
                block_confidence_scores, threshold, mask_id)

            # デバッグ情報
            valid_scores = block_confidence_scores[block_confidence_scores != -np.inf]
            remaining_masks = (
                x[:, current_block_start:current_block_end] == mask_id).sum().item()

            if len(valid_scores) > 0:
                print(f"ブロック {num_block}: 残りマスク={remaining_masks}, "
                      f"信頼度範囲=[{valid_scores.min():.3f}, {valid_scores.max():.3f}], "
                      f"曖昧度スコア={ambiguity_score:.3f}")
        else:
            ambiguity_score = 0.0

        block_confidences.append(ambiguity_score)

    # Phase 2: 動的予算再配分
    t_used_base = t_base * num_blocks
    t_refine = max(0, steps - t_used_base)

    print(f"Phase 2: 予算再配分 - 使用済み: {t_used_base}, 残り予算: {t_refine}")
    print(f"ブロック曖昧度スコア: {block_confidences}")

    # 真の早期終了チェック: 全ブロックが完成済みなら終了
    total_remaining_masks = (x[:, prompt.shape[1]:] == mask_id).sum().item()
    if total_remaining_masks == 0:
        print(f"早期終了: 全ブロック完成済み - 総NFE: {nfe} (削減: {steps - nfe})")
        return x, nfe, block_confidences

    if t_refine > 0:
        # 曖昧度に基づく追加ステップ配分
        additional_steps = allocate_refinement_budget(
            block_confidences, t_refine)
        print(f"追加ステップ配分: {additional_steps}")

        # Phase 3: 改善された標的精錬
        for num_block in range(num_blocks):
            if additional_steps[num_block] == 0:
                continue

            current_block_start = prompt.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            # ブロック完成チェック
            block_masks = (
                x[:, current_block_start:current_block_end] == mask_id).sum().item()
            if block_masks == 0:
                print(f"ブロック {num_block}: すでに完成 - スキップ")
                continue

            print(
                f"ブロック {num_block} の追加精錬中: {additional_steps[num_block]} ステップ")

            # 修正: 再マスキングなしの安定した精錬
            for i in range(additional_steps[num_block]):
                # ブロック完成チェック（真の早期終了）
                current_masks = (
                    x[:, current_block_start:current_block_end] == mask_id).sum().item()
                if current_masks == 0:
                    print(f"  ステップ {i+1}: ブロック完成 - 早期終了")
                    break

                nfe += 1
                mask_index = (x == mask_id)
                logits = model(x).logits
                mask_index[:, current_block_end:] = 0

                # 標準の転送インデックス使用（再マスキングなし）
                x0, transfer_index = get_transfer_index(
                    logits, temperature, remasking, mask_index, x,
                    torch.tensor([[1]], device=x.device), threshold=None)

                x[transfer_index] = x0[transfer_index]

                remaining_masks = (
                    x[:, current_block_start:current_block_end] == mask_id).sum().item()
                print(f"  ステップ {i+1}: 残りマスク={remaining_masks}, NFE={nfe}")

                # NFE上限チェック
                if nfe >= steps:
                    print(f"  NFE上限到達 - 精錬終了")
                    break

            # 全体的なNFE上限チェック
            if nfe >= steps:
                break

    print(f"DRS生成完了 - 総NFE: {nfe}, 削減率: {((steps - nfe) / steps * 100):.1f}%")
    return x, nfe, block_confidences


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
