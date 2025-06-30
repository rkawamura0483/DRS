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

"""
BasicGTSとSemanticGTSの基本テスト（scipy不要）
"""

import torch
import numpy as np
import sys
import os

# パッケージ構造に基づいてimport
from llada.metrics.gts import BasicGTS, SemanticGTS, create_gts_metric


def test_basic_gts_stability():
    """BasicGTSの安定性テスト"""
    print("🧪 BasicGTS安定性テスト開始...")

    metric = BasicGTS()

    # 完全に安定な軌跡
    logits = torch.tensor([[[10.0, 1.0, 1.0], [1.0, 10.0, 1.0]]])

    for step in range(5):
        metric.update(logits, step)

    score = metric.value()
    print(f"   完全安定軌跡のスコア: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"

    # 不安定な軌跡
    metric = BasicGTS()
    logits_sequence = [
        torch.tensor([[[10.0, 1.0, 1.0], [1.0, 10.0, 1.0]]]),  # argmax: [0, 1]
        torch.tensor([[[1.0, 10.0, 1.0], [1.0, 1.0, 10.0]]]),  # argmax: [1, 2]
    ]

    for step, logits in enumerate(logits_sequence):
        metric.update(logits, step)

    score = metric.value()
    print(f"   不安定軌跡のスコア: {score}")
    assert score < 1.0, f"不安定軌跡のスコアは1.0未満であるべき, got {score}"

    print("✅ BasicGTS テスト成功")


def test_semantic_gts():
    """SemanticGTSの基本テスト"""
    print("🧪 SemanticGTS基本テスト開始...")

    # テスト用埋め込み行列
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # token 0
        [0.0, 1.0, 0.0, 0.0],  # token 1 (token 0と直交)
        [1.0, 1.0, 0.0, 0.0],  # token 2 (token 0と類似)
    ], dtype=torch.float32)

    # 変更なしのケース
    metric = SemanticGTS(embeddings)
    logits = torch.tensor([[[10.0, 1.0, 1.0]]])

    for step in range(3):
        metric.update(logits, step)

    score = metric.value()
    print(f"   変更なしのスコア: {score}")
    assert score == 1.0, "変更がない場合は1.0であるべき"

    # 類似トークンへの変更
    metric = SemanticGTS(embeddings)
    logits_sequence = [
        torch.tensor([[[10.0, 1.0, 1.0]]]),  # token 0
        torch.tensor([[[1.0, 1.0, 10.0]]]),  # token 2 (類似)
    ]

    for step, logits in enumerate(logits_sequence):
        metric.update(logits, step)

    similar_score = metric.value()
    print(f"   類似変更のスコア: {similar_score}")

    # 非類似トークンへの変更
    metric = SemanticGTS(embeddings)
    logits_sequence = [
        torch.tensor([[[10.0, 1.0, 1.0]]]),  # token 0
        torch.tensor([[[1.0, 10.0, 1.0]]]),  # token 1 (直交)
    ]

    for step, logits in enumerate(logits_sequence):
        metric.update(logits, step)

    dissimilar_score = metric.value()
    print(f"   非類似変更のスコア: {dissimilar_score}")

    # 類似変更の方がスコアが高いはず
    assert similar_score > dissimilar_score, \
        f"類似変更のスコア({similar_score})は非類似変更({dissimilar_score})より高くあるべき"

    print("✅ SemanticGTS テスト成功")


def test_factory_function():
    """ファクトリ関数のテスト"""
    print("🧪 ファクトリ関数テスト開始...")

    # BasicGTS作成
    basic_metric = create_gts_metric('basic')
    assert isinstance(basic_metric, BasicGTS)
    print("   BasicGTS作成成功")

    # SemanticGTS作成
    embeddings = torch.randn(100, 64)
    semantic_metric = create_gts_metric(
        'semantic', embedding_matrix=embeddings)
    assert isinstance(semantic_metric, SemanticGTS)
    print("   SemanticGTS作成成功")

    # 無効なタイプ
    try:
        create_gts_metric('invalid')
        assert False, "無効なタイプでは例外が発生すべき"
    except ValueError:
        print("   無効タイプの例外処理成功")

    print("✅ ファクトリ関数テスト成功")


def test_token_position_masking():
    """トークン位置マスクのテスト"""
    print("🧪 トークン位置マスクテスト開始...")

    metric = BasicGTS()

    # 3トークンのlogitsだが、最初の2つだけを評価
    logits = torch.tensor(
        [[[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [5.0, 5.0, 5.0]]])
    token_positions = torch.tensor([[True, True, False]])  # 最初の2つのみ

    metric.update(logits, 0, token_positions)

    # 評価対象トークン数が2であることを確認
    assert metric.num_tokens == 2, f"Expected 2 tokens, got {metric.num_tokens}"
    print(f"   マスクされたトークン数: {metric.num_tokens}")

    print("✅ トークン位置マスクテスト成功")


if __name__ == "__main__":
    print("🚀 GTS基本テスト開始...")

    test_basic_gts_stability()
    test_semantic_gts()
    test_factory_function()
    test_token_position_masking()

    print("🎉 全テスト成功！Step-1完了")
