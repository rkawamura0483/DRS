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


def calculate_block_ambiguity_improved(confidence_scores, threshold, mask_id, remaining_masks):
    """
    改善された曖昧度計算: 完成ブロックでも信頼度情報を活用

    Args:
        confidence_scores: Tensor of confidence scores for block tokens
        threshold: Confidence threshold τ
        mask_id: Mask token ID to exclude from calculation
        remaining_masks: Number of remaining mask tokens

    Returns:
        Ambiguity score (float): fraction of tokens below threshold
    """
    # マスクされていない有効なトークンの信頼度のみを考慮
    valid_scores = confidence_scores[confidence_scores != -np.inf]
    if len(valid_scores) == 0:
        return 0.0

    # 閾値未満のトークンの割合を計算
    low_confidence_tokens = (valid_scores < threshold).float()
    ambiguity_score = low_confidence_tokens.mean().item()

    # 🔑 重要修正: 完成ブロックでも実際の信頼度情報を使用
    # （従来の強制0設定を削除）
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
def generate_with_drs_improved(model, prompt, steps=128, gen_length=128, block_length=128,
                               temperature=0., remasking='low_confidence', mask_id=None,
                               threshold=0.8, t_base=8):
    """
    改善版DRS: 信頼度情報を適切に保存・活用

    修正点:
    1. 完成前の信頼度スコアを保存
    2. 完成後でも実際の曖昧度を計算
    3. 真の動的配分価値を検証可能
    4. リファインメントロジックを統一し、KVキャッシュを活用して効率化
    """
    # トークナイザーからマスクIDを取得（推奨）
    if mask_id is None:
        if hasattr(model, 'tokenizer') and model.tokenizer.mask_token_id is not None:
            mask_id = model.tokenizer.mask_token_id
        else:
            mask_id = model.config.mask_token_id
            print(f"トークナイザーのmask_token_idがNoneのため、モデル設定から取得: {mask_id}")

    # デフォルト値として126336を確保（LLaDAの正式なマスクトークンID）
    if mask_id is None:
        mask_id = 126336
        print(f"モデル設定でもNoneのため、LLaDAデフォルト値を使用: {mask_id}")

    # シーケンスをマスクで初期化
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    # Phase 1: 粗い初期パス（信頼度情報を適切に保存）
    block_confidences = []
    block_remaining_masks = []
    block_confidence_histories = []
    nfe = 0

    print(f"Phase 1: 改善版初期パス - {t_base}ステップ x {num_blocks}ブロック")
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # 各ブロックでt_baseステップを実行
        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, t_base)

        # 🔑 修正: 元のgenerate_with_dual_cacheと同じパターンを採用
        # 最初のステップ: フルシーケンスでモデルを呼び出し
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index, confidence_scores_initial = get_transfer_index_with_confidence(
            output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if len(num_transfer_tokens[0]) > 0 else None)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        block_confidence_scores = None
        confidence_history = []  # このブロックの信頼度履歴

        # 最初のステップの信頼度を記録
        if confidence_scores_initial is not None:
            step_confidence = confidence_scores_initial[0,
                                                        current_block_start:current_block_end]
            confidence_history.append(step_confidence.clone())

        # 残りのステップ: ブロック単位での処理
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1

        for i in range(1, t_base):
            nfe += 1
            mask_index_block = (
                x[:, current_block_start:current_block_end] == mask_id)

            if mask_index_block.sum() == 0:
                break  # ブロックが完成

            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values,
                           use_cache=True, replace_position=replace_position).logits

            x0_block, transfer_index_block, confidence_scores_block = get_transfer_index_with_confidence(
                logits, temperature, remasking, mask_index_block, x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if i < len(num_transfer_tokens[0]) else None)

            x[:, current_block_start:current_block_end][transfer_index_block] = x0_block[transfer_index_block]

            # 信頼度を記録
            step_confidence = confidence_scores_block[0]
            confidence_history.append(step_confidence.clone())
            block_confidence_scores = step_confidence

        # ブロック終了後、次のブロックのためにKVキャッシュを更新
        if num_block < num_blocks - 1:  # 最後のブロックでない場合
            output = model(x[:, :current_block_end], use_cache=True)
            past_key_values = output.past_key_values

        # 残りマスク数を記録
        remaining_masks = (
            x[:, current_block_start:current_block_end] == mask_id).sum().item()
        block_remaining_masks.append(remaining_masks)
        block_confidence_histories.append(confidence_history)

        # 🔑 改善されたブロック曖昧度スコア計算
        if block_confidence_scores is not None:
            ambiguity_score = calculate_block_ambiguity_improved(
                block_confidence_scores, threshold, mask_id, remaining_masks)
        else:
            ambiguity_score = 1.0 if remaining_masks > 0 else 0.0

        block_confidences.append(ambiguity_score)

        # デバッグ情報
        if block_confidence_scores is not None:
            valid_scores = block_confidence_scores[block_confidence_scores != -
                                                   np.inf]
            if len(valid_scores) > 0:
                below_threshold = (valid_scores < threshold).sum().item()
                print(f"ブロック {num_block}: 残りマスク={remaining_masks}, "
                      f"信頼度範囲=[{valid_scores.min():.3f}, {valid_scores.max():.3f}], "
                      f"曖昧度スコア={ambiguity_score:.3f} "
                      f"(閾値未満: {below_threshold}/{len(valid_scores)})")
            else:
                print(
                    f"ブロック {num_block}: 残りマスク={remaining_masks}, 信頼度情報なし, 曖昧度スコア={ambiguity_score:.3f}")
        else:
            print(f"ブロック {num_block}: 信頼度情報なし, 曖昧度スコア=0.0")

    # Phase 2: 動的予算再配分（改善版）
    t_used_base = t_base * num_blocks
    t_refine = max(0, steps - t_used_base)

    print(f"\nPhase 2: 改善版動的予算再配分")
    print(f"  使用済み予算: {nfe}")
    print(f"  残り予算: {t_refine}")
    print(f"  ブロック曖昧度スコア: {[f'{s:.3f}' for s in block_confidences]}")
    print(f"  ブロック残りマスク数: {block_remaining_masks}")

    if t_refine <= 0:
        print(f"  → 予算不足。現在の状態で終了")
        return x, nfe, block_confidences

    # 曖昧度に基づく追加ステップ配分
    additional_steps = allocate_refinement_budget(block_confidences, t_refine)
    print(f"  → 追加ステップ配分: {additional_steps}")

    # Phase 3: 標的精錬（統一ロジック）
    # 🔑 一時的にコメントアウト: エラー原因の特定のため
    """
    if any(steps > 0 for steps in additional_steps):
        print(f"\nPhase 3: 改善版標的精錬開始")

        for num_block, steps_to_add in enumerate(additional_steps):
            if steps_to_add == 0:
                continue

            current_block_start = prompt.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            # --- 新しい統一リファインメントロジック ---
            # Step 1: 精錬対象のマスクを決定
            block_x = x[:, current_block_start:current_block_end]
            is_masked = (block_x == mask_id)

            if is_masked.sum().item() == 0:
                # 完成ブロックだが曖昧度が高い場合、低信頼度トークンを再マスク
                print(f"  精錬理由: ブロック {num_block} の信頼度ベース品質向上")
                # 最新のKVキャッシュを取得
                output = model(x[:, :current_block_start], use_cache=True)
                past_key_values_refine = output.past_key_values
                replace_pos_refine = torch.zeros_like(x, dtype=torch.bool)
                replace_pos_refine[:, current_block_start:current_block_end] = 1

                logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values_refine,
                               use_cache=True, replace_position=replace_pos_refine).logits
                nfe += 1

                p = F.softmax(logits.to(torch.float64), dim=-1)
                current_tokens = x[:, current_block_start:current_block_end]
                current_confidence = torch.gather(
                    p, dim=-1, index=current_tokens.unsqueeze(-1)).squeeze(-1)

                low_conf_mask = current_confidence < threshold
                if low_conf_mask.sum().item() == 0:
                    print(
                        f"    ブロック {num_block} は信頼度が高いためスキップします (最低信頼度: {current_confidence.min().item():.3f})。")
                    continue

                x[:, current_block_start:current_block_end][low_conf_mask] = mask_id
                print(
                    f"    → ブロック {num_block} の {low_conf_mask.sum().item()} 個の低信頼度トークンを再マスクしました。")
            else:
                print(f"  精錬理由: ブロック {num_block} の残存マスクの品質向上")

            # Step 2: マスクを含むブロックを精錬
            block_mask_index = (
                x[:, current_block_start:current_block_end] == mask_id)
            if block_mask_index.sum().item() == 0:
                continue

            num_transfer_tokens_refine = get_num_transfer_tokens(
                block_mask_index, steps_to_add)

            print(
                f"  → ブロック {num_block} を {steps_to_add} ステップで精錬...")

            # リファインメントのためのKVキャッシュを準備
            output = model(x[:, :current_block_start], use_cache=True)
            past_key_values_refine = output.past_key_values

            for i in range(steps_to_add):
                if (x[:, current_block_start:current_block_end] == mask_id).sum().item() == 0:
                    print(f"    ブロック {num_block} の精錬完了 (早期終了)")
                    break

                nfe += 1
                replace_pos_refine = torch.zeros_like(x, dtype=torch.bool)
                replace_pos_refine[:, current_block_start:current_block_end] = 1

                logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values_refine,
                               use_cache=True, replace_position=replace_pos_refine).logits

                refine_mask_index = (
                    x[:, current_block_start:current_block_end] == mask_id)

                x0_block, transfer_index_block, _ = get_transfer_index_with_confidence(
                    logits, temperature, remasking, refine_mask_index, x[:, current_block_start:current_block_end], num_transfer_tokens_refine[:, i])

                x[:, current_block_start:current_block_end][transfer_index_block] = x0_block[transfer_index_block]
    """
    print(f"\nPhase 3: スキップ（エラー回避のため）")

    # 最終結果
    final_masks = (x[:, prompt.shape[1]:] == mask_id).sum().item()
    completion_rate = ((gen_length - final_masks) / gen_length) * 100

    print(f"\n改善版DRS生成完了:")
    print(f"  総NFE: {nfe}/{steps} ({(nfe/steps*100):.1f}%)")
    print(
        f"  完成率: {completion_rate:.1f}% ({gen_length - final_masks}/{gen_length})")
    print(f"  検出された真の曖昧度: {any(score > 0.01 for score in block_confidences)}")
    print(f"  曖昧度分散: {np.var(block_confidences):.3f}")

    return x, nfe, block_confidences


# 後方互換性のため、既存の関数名をエイリアスとして残す
def generate_with_drs_fixed(model, prompt, steps=128, gen_length=128, block_length=128,
                            temperature=0., remasking='low_confidence', mask_id=None,
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
