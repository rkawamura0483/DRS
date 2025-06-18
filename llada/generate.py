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
             remasking='low_confidence', mask_id=None, threshold=None):
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
        mask_id: The toke id of [MASK]. If None, will be obtained from model config.
    '''
    # mask_idを適切に取得
    if mask_id is None:
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'mask_token_id') and model.tokenizer.mask_token_id is not None:
            mask_id = model.tokenizer.mask_token_id
        else:
            mask_id = model.config.mask_token_id

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
                               remasking='low_confidence', mask_id=None, threshold=None):
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
        mask_id: The toke id of [MASK]. If None, will be obtained from model config.
    '''
    # mask_idを適切に取得
    if mask_id is None:
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'mask_token_id') and model.tokenizer.mask_token_id is not None:
            mask_id = model.tokenizer.mask_token_id
        else:
            mask_id = model.config.mask_token_id

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
                             remasking='low_confidence', mask_id=None, threshold=None):
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
        mask_id: The toke id of [MASK]. If None, will be obtained from model config.
    '''
    # mask_idを適切に取得
    if mask_id is None:
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'mask_token_id') and model.tokenizer.mask_token_id is not None:
            mask_id = model.tokenizer.mask_token_id
        else:
            mask_id = model.config.mask_token_id

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


def _calculate_ambiguity(confidence_scores, threshold):
    """
    ブロックの曖昧度を計算する。
    信頼度がしきい値を下回るトークンの割合として定義する。

    Args:
        confidence_scores (Tensor): ブロック内のトークンの信頼度スコア。
        threshold (float): 信頼度のしきい値。

    Returns:
        float: 曖昧度スコア (0.0 - 1.0)。
    """
    # マスクされておらず、有効な信頼度スコアのみを対象とする
    valid_scores = confidence_scores[confidence_scores != -np.inf]
    if valid_scores.numel() == 0:
        return 0.0  # 有効なトークンがない場合、曖昧度はない

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
        # No ambiguity, distribute equally if there's any masks left to refine
        # This part might not be strictly needed if ambiguity is calculated on final blocks
        steps_per_block = total_refinement_budget // len(block_ambiguities)
        remainder = total_refinement_budget % len(block_ambiguities)
        additional_steps = [steps_per_block] * len(block_ambiguities)
        for i in range(remainder):
            additional_steps[i] += 1
        return additional_steps

    # Proportional allocation
    additional_steps = [0] * len(block_ambiguities)
    for i in range(total_refinement_budget):
        proportions = [amb / total_ambiguity for amb in block_ambiguities]
        most_ambiguous_block = proportions.index(max(proportions))
        additional_steps[most_ambiguous_block] += 1
        # To avoid one block taking all, we can reduce its ambiguity after allocation
        # For simplicity, we stick to proportional allocation of the total budget at once.

    proportions = [amb / total_ambiguity for amb in block_ambiguities]
    additional_steps = [round(p * total_refinement_budget)
                        for p in proportions]

    # Adjust to match total budget due to rounding
    current_total = sum(additional_steps)
    diff = total_refinement_budget - current_total
    if diff != 0:
        # Distribute remainder/deficit to most/least ambiguous blocks
        sort_order = sorted(range(len(block_ambiguities)),
                            key=lambda k: block_ambiguities[k], reverse=(diff > 0))
        for i in range(abs(diff)):
            additional_steps[sort_order[i %
                                        len(sort_order)]] += (1 if diff > 0 else -1)

    return additional_steps


@torch.no_grad()
def generate_with_drs(model, prompt, steps=128, gen_length=128, block_length=128,
                      temperature=0., remasking='low_confidence', mask_id=None,
                      threshold=0.8, t_base=8):
    """
    Dynamic Refinement Steps (DRS) を用いた生成関数。
    研究仮説をクリーンに検証するために再実装されたバージョン。

    処理は3つのフェーズで構成される:
    1. 初期生成: 各ブロックを固定ステップ数(t_base)で生成し、信頼度情報を収集する。
    2. 予算配分: 各ブロックの曖昧度を計算し、残りの推論予算を動的に配分する。
    3. 動的リファインメント: 予算が配分されたブロックに対し、低信頼度トークンを再マスクして品質を向上させる。
    """
    # マスクIDの取得
    if mask_id is None:
        if hasattr(model, 'tokenizer') and model.tokenizer.mask_token_id is not None:
            mask_id = model.tokenizer.mask_token_id
        else:
            mask_id = model.config.mask_token_id or 126336

    # シーケンスをマスクで初期化
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    nfe = 0
    block_final_confidences = []

    # ======== フェーズ1: 初期生成 ========
    print(f"Phase 1: 初期生成 (t_base={t_base} per block)")

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # generate_with_dual_cache を参考にした、効率的なKVキャッシュ戦略
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        nfe += 1

        # 最初のステップ: 全体シーケンスからlogitsを取得し、マスクを更新
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0  # 現在のブロック以降はマスクしない
        num_initial_transfer = get_num_transfer_tokens(mask_index, t_base)[
            :, 0]
        x0, transfer_index, _ = get_transfer_index_with_confidence(
            output.logits, temperature, remasking, mask_index, x, num_initial_transfer
        )
        x[transfer_index] = x0[transfer_index]

        # KVキャッシュをプレフィックス部分のみに刈り込む
        trimmed_past_key_values = []
        for i in range(len(past_key_values)):
            trimmed_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                trimmed_past_key_values[i] += (past_key_values[i]
                                               [j][:, :, :current_block_start],)

        # 残りの t_base-1 ステップを現在のブロックで実行
        for i in range(1, t_base):
            mask_index_block = (
                x[:, current_block_start:current_block_end] == mask_id)
            if mask_index_block.sum() == 0:
                print(f"  ブロック {num_block}: {i} ステップで早期完了")
                break

            replace_position = torch.zeros(
                (1, block_length), dtype=torch.bool, device=x.device)
            replace_position[:, :] = 1  # ブロック全体を対象

            logits = model(x[:, current_block_start:current_block_end], past_key_values=trimmed_past_key_values,
                           use_cache=True, replace_position=replace_position).logits
            nfe += 1

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index_block, t_base - i)[:, 0]
            x0_block, transfer_index_block, _ = get_transfer_index_with_confidence(
                logits, temperature, remasking, mask_index_block,
                x[:, current_block_start:current_block_end], num_transfer_tokens
            )

            x[:, current_block_start:current_block_end][transfer_index_block] = x0_block[transfer_index_block]

        # 最終的なブロックの信頼度を計算・保存
        final_logits = model(
            x[:, :current_block_end], use_cache=False).logits[:, current_block_start:current_block_end]
        nfe += 1
        p = F.softmax(final_logits.to(torch.float64), dim=-1)
        final_tokens = x[:, current_block_start:current_block_end]
        final_confidence = torch.gather(
            p, dim=-1, index=final_tokens.unsqueeze(-1)).squeeze(-1)
        block_final_confidences.append(final_confidence[0])

        # 次のブロックのためにKVキャッシュを更新
        if num_block < num_blocks - 1:
            output = model(x[:, :current_block_end], use_cache=True)
            past_key_values = output.past_key_values
            nfe += 1

    # ======== フェーズ2: 予算配分 ========
    t_used_base = nfe
    t_refine = max(0, steps - t_used_base)
    print(f"\nPhase 2: 予算配分 (残り予算: {t_refine})")

    block_ambiguities = [_calculate_ambiguity(
        conf, threshold) for conf in block_final_confidences]
    print(f"  ブロック曖昧度スコア: {[f'{s:.3f}' for s in block_ambiguities]}")

    if t_refine <= 0 or sum(block_ambiguities) == 0:
        print("  → リファインメント予算なし、または全ブロック高品質のため終了")
        return x, nfe, block_ambiguities

    additional_steps = allocate_refinement_budget(block_ambiguities, t_refine)
    print(f"  → 追加ステップ配分: {additional_steps}")

    # ======== フェーズ3: 動的リファインメント ========
    if any(s > 0 for s in additional_steps):
        print("\nPhase 3: 動的リファインメント開始")

        for num_block, steps_to_add in enumerate(additional_steps):
            if steps_to_add == 0:
                continue

            print(f"  → ブロック {num_block} を {steps_to_add} ステップでリファインメント")
            current_block_start = prompt.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            # --- 低信頼度トークンの再マスク ---
            current_confidence = block_final_confidences[num_block]
            remask_index = current_confidence < threshold
            num_remasked = remask_index.sum().item()

            if num_remasked > 0:
                print(f"    - {num_remasked}個の低信頼度トークンを再マスク")
                x[:, current_block_start:current_block_end][remask_index.unsqueeze(
                    0)] = mask_id
            else:
                print(f"    - 再マスク対象なし、高品質のためスキップ")
                continue

            # --- リファインメント実行 ---
            # KVキャッシュの準備
            replace_position_refine = torch.zeros(
                (1, current_block_start), dtype=torch.bool, device=x.device)
            output = model(x[:, :current_block_start], use_cache=True,
                           replace_position=replace_position_refine)
            past_key_values_refine = output.past_key_values
            nfe += 1

            for i in range(steps_to_add):
                mask_index_block = (
                    x[:, current_block_start:current_block_end] == mask_id)
                if mask_index_block.sum() == 0:
                    print(f"    - {i} ステップでリファインメント完了")
                    break

                nfe += 1
                replace_position = torch.zeros(
                    (1, block_length), dtype=torch.bool, device=x.device)
                replace_position[:, :] = 1

                logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values_refine,
                               use_cache=True, replace_position=replace_position).logits

                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index_block, steps_to_add - i)
                x0_block, transfer_index_block, _ = get_transfer_index_with_confidence(
                    logits, temperature, remasking, mask_index_block,
                    x[:, current_block_start:current_block_end], num_transfer_tokens[:, 0]
                )
                x[:, current_block_start:current_block_end][transfer_index_block] = x0_block[transfer_index_block]

    final_masks = (x[:, prompt.shape[1]:] == mask_id).sum().item()
    print(f"\nDRS生成完了: 総NFE={nfe}, 未生成トークン数={final_masks}")
    return x, nfe, block_ambiguities


def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    model.tokenizer = tokenizer  # モデルにトークナイザーをセット

    # mask_idを確実に取得
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = model.config.mask_token_id
        print(f"トークナイザーのmask_token_idがNoneのため、モデル設定から取得: {mask_id}")

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out, nfe = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128,
                                        block_length=32, temperature=0., remasking='low_confidence', mask_id=mask_id)
    print(tokenizer.batch_decode(
        out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
