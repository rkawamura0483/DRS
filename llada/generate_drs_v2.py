# 理論的改善版DRS実装
import numpy as np
import torch
import torch.nn.functional as F
from generate import add_gumbel_noise, get_num_transfer_tokens, get_transfer_index


@torch.no_grad()
def generate_with_improved_drs(model, prompt, steps=128, gen_length=128, block_length=128,
                               temperature=0., remasking='low_confidence', mask_id=None,
                               threshold=0.8, t_base=8, adaptive_threshold=True):
    """
    理論的改善版DRS実装
    - 動的信頼度評価（ブロック完了前）
    - 未完了ブロック優先予算配分
    - 文脈継続性保持メカニズム
    """

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

    nfe = 0
    block_states = []  # ブロック状態：完了度、信頼度、マスク数

    print(
        f"📊 改善版DRS開始 (t_base={t_base}, adaptive_threshold={adaptive_threshold})")

    # ======== フェーズ1: 動的初期生成 ========
    print(f"\nPhase 1: 動的初期生成")

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, t_base)

        # ブロック初期化
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(
            output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0])
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        # ブロック内適応的生成
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1

        confidence_history = []

        for i in range(1, t_base):
            nfe += 1
            mask_index = (
                x[:, current_block_start:current_block_end] == mask_id)
            if mask_index.sum() == 0:
                break  # ブロック完了

            logits = model(x[:, current_block_start:current_block_end],
                           past_key_values=past_key_values,
                           use_cache=True, replace_position=replace_position).logits

            # **改善点1: リアルタイム信頼度評価**
            p = F.softmax(logits.to(torch.float64), dim=-1)
            current_tokens = x[:, current_block_start:current_block_end]
            confidence = torch.gather(
                p, dim=-1, index=current_tokens.unsqueeze(-1)).squeeze(-1)
            confidence_history.append(confidence[0])

            # **改善点2: 適応的閾値**
            if adaptive_threshold and len(confidence_history) > 2:
                recent_confidence = torch.stack(confidence_history[-3:])
                dynamic_threshold = threshold * \
                    (1 + 0.1 * recent_confidence.std().item())
            else:
                dynamic_threshold = threshold

            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index,
                                                    x[:, current_block_start:current_block_end],
                                                    num_transfer_tokens[:, i])
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]

        # ブロック状態記録
        remaining_masks = (
            x[:, current_block_start:current_block_end] == mask_id).sum().item()
        avg_confidence = torch.stack(confidence_history).mean(
        ).item() if confidence_history else 0.5

        block_states.append({
            'block_id': num_block,
            'completion': 1.0 - (remaining_masks / block_length),
            'avg_confidence': avg_confidence,
            'remaining_masks': remaining_masks,
            'confidence_history': confidence_history
        })

        print(f"  ブロック {num_block}: 完了度 {block_states[-1]['completion']:.2f}, "
              f"平均信頼度 {avg_confidence:.3f}, 残りマスク {remaining_masks}")

    # ======== フェーズ2: インテリジェント予算配分 ========
    t_used = nfe
    t_remaining = max(0, steps - t_used)
    print(f"\nPhase 2: インテリジェント予算配分 (使用済み: {t_used}, 残り: {t_remaining})")

    if t_remaining <= 0:
        print("  → 予算なし、初期生成で終了")
        return x, nfe, [bs['avg_confidence'] for bs in block_states]

    # **改善点3: 未完了ブロック優先配分**
    incomplete_blocks = [bs for bs in block_states if bs['completion'] < 1.0]

    if not incomplete_blocks:
        print("  → 全ブロック完了済み、低信頼度ブロックを再処理")
        # 全完了の場合、信頼度最低ブロックを選択
        target_blocks = sorted(block_states, key=lambda x: x['avg_confidence'])[
            :min(3, len(block_states))]
    else:
        print(f"  → {len(incomplete_blocks)}個の未完了ブロックを優先処理")
        # 未完了ブロックを効率重視で配分
        target_blocks = sorted(incomplete_blocks,
                               key=lambda x: x['remaining_masks'] /
                               max(x['avg_confidence'], 0.1),
                               reverse=True)

    # 予算配分計算
    budget_allocation = {}
    remaining_budget = t_remaining

    for i, block in enumerate(target_blocks):
        if remaining_budget <= 0:
            break

        # 未完了ブロックには多めに配分
        if block['completion'] < 1.0:
            allocated = min(remaining_budget, block['remaining_masks'] * 2)
        else:
            # 完了済み低信頼度ブロックには少なめ配分
            allocated = min(remaining_budget // 2, max(1,
                            int(block_length * (1 - block['avg_confidence']))))

        budget_allocation[block['block_id']] = allocated
        remaining_budget -= allocated

    print(f"  → 予算配分: {budget_allocation}")

    # ======== フェーズ3: 文脈継続性考慮リファインメント ========
    if budget_allocation:
        print(f"\nPhase 3: 文脈継続性考慮リファインメント")

        for block_id, extra_steps in budget_allocation.items():
            if extra_steps <= 0:
                continue

            print(f"  → ブロック {block_id}: {extra_steps} ステップでリファインメント")
            current_block_start = prompt.shape[1] + block_id * block_length
            current_block_end = current_block_start + block_length

            block_state = block_states[block_id]

            # **改善点4: 選択的再マスク**（完了ブロックのみ）
            if block_state['completion'] >= 1.0:
                # 信頼度履歴から最も不安定な位置を特定
                if block_state['confidence_history']:
                    final_confidence = block_state['confidence_history'][-1]
                    remask_threshold = threshold + 0.1  # より厳格な閾値
                    remask_indices = final_confidence < remask_threshold
                    num_remasked = remask_indices.sum().item()

                    if num_remasked > 0:
                        print(f"    - {num_remasked}個のトークンを選択的再マスク")
                        x[:, current_block_start:current_block_end][remask_indices.unsqueeze(
                            0)] = mask_id
                    else:
                        print(f"    - 再マスク対象なし、スキップ")
                        continue

            # **改善点5: 文脈考慮リファインメント**
            # 隣接ブロックとの整合性を考慮
            context_start = max(
                prompt.shape[1], current_block_start - block_length // 2)
            context_end = min(
                x.shape[1], current_block_end + block_length // 2)

            output = model(x[:, context_start:context_end], use_cache=True)
            # ターゲットブロック部分のlogitsのみを使用
            target_offset = current_block_start - context_start
            target_logits = output.logits[:,
                                          target_offset:target_offset + block_length]

            past_key_values = output.past_key_values
            nfe += 1

            for step in range(extra_steps):
                mask_index = (
                    x[:, current_block_start:current_block_end] == mask_id)
                if mask_index.sum() == 0:
                    print(f"    - {step+1} ステップで完了")
                    break

                nfe += 1

                # 文脈考慮での予測
                context_output = model(x[:, context_start:current_block_end],
                                       past_key_values=past_key_values, use_cache=True)
                logits = context_output.logits[:, target_offset:]

                remaining_steps = extra_steps - step
                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index, remaining_steps)[:, 0]

                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index,
                                                        x[:, current_block_start:current_block_end],
                                                        num_transfer_tokens)
                x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]

    final_masks = (x[:, prompt.shape[1]:] == mask_id).sum().item()
    print(f"\n改善版DRS完了: 総NFE={nfe}, 未生成トークン={final_masks}")

    # 最終品質指標
    final_confidences = []
    for bs in block_states:
        if bs['confidence_history']:
            final_confidences.append(
                bs['confidence_history'][-1].mean().item())
        else:
            final_confidences.append(bs['avg_confidence'])

    return x, nfe, final_confidences


def compare_drs_versions(model, input_ids, **kwargs):
    """DRS版数比較実験"""
    print("🔬 DRS版数比較実験")

    from generate import generate_with_drs, generate_with_dual_cache

    # オリジナルfast-dLLM
    print("\n1️⃣ オリジナルfast-dLLM")
    baseline_out, baseline_nfe = generate_with_dual_cache(
        model, input_ids, **kwargs)

    # 元のDRS
    print("\n2️⃣ 元のDRS")
    original_drs_out, original_drs_nfe, original_ambiguity = generate_with_drs(
        model, input_ids, **kwargs)

    # 改善版DRS
    print("\n3️⃣ 改善版DRS")
    improved_drs_out, improved_drs_nfe, improved_confidences = generate_with_improved_drs(
        model, input_ids, **kwargs)

    print(f"\n📊 比較結果:")
    print(f"  Baseline NFE: {baseline_nfe}")
    print(
        f"  元DRS NFE: {original_drs_nfe} ({original_drs_nfe/baseline_nfe:.2f}x)")
    print(
        f"  改善DRS NFE: {improved_drs_nfe} ({improved_drs_nfe/baseline_nfe:.2f}x)")

    return {
        'baseline': (baseline_out, baseline_nfe),
        'original_drs': (original_drs_out, original_drs_nfe, original_ambiguity),
        'improved_drs': (improved_drs_out, improved_drs_nfe, improved_confidences)
    }
