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
GTSメトリクスのユニットテスト

各GTSメトリクスの動作を人工データで検証します。
"""

from llada.metrics.gts import (
    BasicGTS, SemanticGTS, ProbabilisticGTS,
    create_gts_metric, evaluate_trajectory_stability
)
import pytest
import torch
import numpy as np
import sys
import os

# パッケージ構造に基づいてimport


class TestBasicGTS:
    """BasicGTSのテストクラス"""

    def test_perfect_stability(self):
        """完全に安定な軌跡のテスト"""
        metric = BasicGTS()

        # 同じlogitsを複数ステップ入力
        logits = torch.tensor(
            [[[10.0, 1.0, 1.0], [1.0, 10.0, 1.0]]])  # (1, 2, 3)

        for step in range(5):
            metric.update(logits, step)

        score = metric.value()
        assert score == 1.0, f"完全安定時のスコアは1.0であるべき, got {score}"

    def test_complete_instability(self):
        """完全に不安定な軌跡のテスト"""
        metric = BasicGTS()
        vocab_size = 3
        num_tokens = 2

        # 毎ステップ全トークンが変わるlogitsを作成
        logits_sequence = [
            # argmax: [0, 1]
            torch.tensor([[[10.0, 1.0, 1.0], [1.0, 10.0, 1.0]]]),
            # argmax: [1, 2]
            torch.tensor([[[1.0, 10.0, 1.0], [1.0, 1.0, 10.0]]]),
            # argmax: [2, 0]
            torch.tensor([[[1.0, 1.0, 10.0], [10.0, 1.0, 1.0]]]),
        ]

        for step, logits in enumerate(logits_sequence):
            metric.update(logits, step)

        score = metric.value()
        # 2トークン × 2ステップ遷移 = 4回の変更全てが発生
        expected_flips = 4
        max_possible_flips = num_tokens * \
            (len(logits_sequence) - 1)  # 2 * 2 = 4
        expected_score = 1.0 - \
            (expected_flips / max_possible_flips)  # 1.0 - 1.0 = 0.0

        assert abs(
            score - expected_score) < 1e-6, f"Expected {expected_score}, got {score}"

    def test_partial_instability(self):
        """部分的に不安定な軌跡のテスト"""
        metric = BasicGTS()

        # 1つのトークンだけが変わるケース
        logits_sequence = [
            # argmax: [0, 1]
            torch.tensor([[[10.0, 1.0, 1.0], [1.0, 10.0, 1.0]]]),
            # argmax: [1, 1] - 1つだけ変更
            torch.tensor([[[1.0, 10.0, 1.0], [1.0, 10.0, 1.0]]]),
        ]

        for step, logits in enumerate(logits_sequence):
            metric.update(logits, step)

        score = metric.value()
        # 1変更 / 2可能 = 0.5, score = 1 - 0.5 = 0.5
        expected_score = 0.5
        assert abs(
            score - expected_score) < 1e-6, f"Expected {expected_score}, got {score}"


class TestSemanticGTS:
    """SemanticGTSのテストクラス"""

    @pytest.fixture
    def sample_embedding_matrix(self):
        """テスト用の埋め込み行列"""
        # 3つのトークンの埋め込み（次元=4）
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # token 0
            [0.0, 1.0, 0.0, 0.0],  # token 1 (token 0と直交)
            [1.0, 1.0, 0.0, 0.0],  # token 2 (token 0と類似)
        ], dtype=torch.float32)
        return embeddings

    def test_no_change_stability(self, sample_embedding_matrix):
        """変更がない場合の安定性テスト"""
        metric = SemanticGTS(sample_embedding_matrix)

        # 同じlogitsを複数回
        logits = torch.tensor([[[10.0, 1.0, 1.0]]])  # argmax: [0]

        for step in range(3):
            metric.update(logits, step)

        score = metric.value()
        assert score == 1.0, "変更がない場合はスコア1.0であるべき"

    def test_semantic_similar_change(self, sample_embedding_matrix):
        """意味的に類似したトークンへの変更テスト"""
        metric = SemanticGTS(sample_embedding_matrix)

        # token 0 -> token 2 への変更（類似度が高い）
        logits_sequence = [
            torch.tensor([[[10.0, 1.0, 1.0]]]),  # argmax: [0]
            torch.tensor([[[1.0, 1.0, 10.0]]]),  # argmax: [2]
        ]

        for step, logits in enumerate(logits_sequence):
            metric.update(logits, step)

        score = metric.value()
        # コサイン類似度が高いので、GTSスコアも高くなるべき
        assert score > 0.5, f"類似トークン変更時のスコアは高くあるべき, got {score}"

    def test_semantic_dissimilar_change(self, sample_embedding_matrix):
        """意味的に異なるトークンへの変更テスト"""
        metric = SemanticGTS(sample_embedding_matrix)

        # token 0 -> token 1 への変更（直交＝類似度0）
        logits_sequence = [
            torch.tensor([[[10.0, 1.0, 1.0]]]),  # argmax: [0]
            torch.tensor([[[1.0, 10.0, 1.0]]]),  # argmax: [1]
        ]

        for step, logits in enumerate(logits_sequence):
            metric.update(logits, step)

        score = metric.value()
        # コサイン類似度が0なので、GTSスコアは低くなるべき
        assert score < 0.6, f"非類似トークン変更時のスコアは低くあるべき, got {score}"


class TestProbabilisticGTS:
    """ProbabilisticGTSのテストクラス"""

    def test_identical_distributions(self):
        """同一分布の場合のテスト"""
        try:
            metric = ProbabilisticGTS()
        except ImportError:
            pytest.skip("scipy not available")

        # 同じlogitsを使用
        logits = torch.tensor([[[2.0, 1.0, 0.5]]])

        for step in range(3):
            metric.update(logits, step)

        score = metric.value()
        assert score == 1.0, "同一分布の場合はスコア1.0であるべき"

    def test_gradually_changing_distributions(self):
        """徐々に変化する分布のテスト"""
        try:
            metric = ProbabilisticGTS()
        except ImportError:
            pytest.skip("scipy not available")

        # 段階的に変化するlogits
        logits_sequence = [
            torch.tensor([[[3.0, 1.0, 0.5]]]),
            torch.tensor([[[2.5, 1.2, 0.8]]]),
            torch.tensor([[[2.0, 1.5, 1.0]]]),
        ]

        for step, logits in enumerate(logits_sequence):
            metric.update(logits, step)

        score = metric.value()
        # 徐々に変化するので中程度の安定性
        assert 0.3 < score < 0.9, f"徐々に変化する分布のスコアは中程度であるべき, got {score}"

    def test_drastically_changing_distributions(self):
        """激しく変化する分布のテスト"""
        try:
            metric = ProbabilisticGTS()
        except ImportError:
            pytest.skip("scipy not available")

        # 激しく変化するlogits
        logits_sequence = [
            torch.tensor([[[10.0, 0.0, 0.0]]]),  # ほぼ確実にtoken 0
            torch.tensor([[[0.0, 10.0, 0.0]]]),  # ほぼ確実にtoken 1
            torch.tensor([[[0.0, 0.0, 10.0]]]),  # ほぼ確実にtoken 2
        ]

        for step, logits in enumerate(logits_sequence):
            metric.update(logits, step)

        score = metric.value()
        # 激しい変化なので低い安定性
        assert score < 0.3, f"激しい変化の分布のスコアは低くあるべき, got {score}"


class TestGTSFactory:
    """GTSファクトリ関数のテスト"""

    def test_create_basic_gts(self):
        """BasicGTS作成テスト"""
        metric = create_gts_metric('basic')
        assert isinstance(metric, BasicGTS)

    def test_create_semantic_gts(self):
        """SemanticGTS作成テスト"""
        embedding_matrix = torch.randn(100, 64)
        metric = create_gts_metric(
            'semantic', embedding_matrix=embedding_matrix)
        assert isinstance(metric, SemanticGTS)

    def test_create_probabilistic_gts(self):
        """ProbabilisticGTS作成テスト"""
        try:
            metric = create_gts_metric('probabilistic')
            assert isinstance(metric, ProbabilisticGTS)
        except ImportError:
            pytest.skip("scipy not available")

    def test_invalid_metric_type(self):
        """無効なメトリクスタイプのテスト"""
        with pytest.raises(ValueError):
            create_gts_metric('invalid_type')


class TestTrajectoryEvaluation:
    """軌跡評価ユーティリティのテスト"""

    def test_evaluate_trajectory_basic(self):
        """基本的な軌跡評価テスト"""
        logits_sequence = [
            torch.tensor([[[2.0, 1.0, 0.5]]]),
            torch.tensor([[[2.0, 1.0, 0.5]]]),
            torch.tensor([[[2.0, 1.0, 0.5]]]),
        ]

        result = evaluate_trajectory_stability(
            logits_sequence,
            metric_type='basic'
        )

        assert 'gts_score' in result
        assert 'metric_type' in result
        assert 'num_steps' in result
        assert 'num_tokens' in result
        assert result['metric_type'] == 'basic'
        assert result['num_steps'] == 3
        assert result['gts_score'] == 1.0  # 完全安定


def test_token_position_masking():
    """トークン位置マスクのテスト"""
    metric = BasicGTS()

    # 3トークンのlogitsだが、最初の2つだけを評価
    logits = torch.tensor(
        [[[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [5.0, 5.0, 5.0]]])
    token_positions = torch.tensor([[True, True, False]])  # 最初の2つのみ

    metric.update(logits, 0, token_positions)

    # 評価対象トークン数が2であることを確認
    assert metric.num_tokens == 2, f"Expected 2 tokens, got {metric.num_tokens}"


if __name__ == "__main__":
    # 簡単な動作確認
    print("🧪 GTSメトリクステスト開始...")

    # BasicGTSの簡単なテスト
    print("📊 BasicGTS テスト...")
    metric = BasicGTS()
    logits = torch.tensor([[[10.0, 1.0], [1.0, 10.0]]])
    metric.update(logits, 0)
    print(f"   初期スコア: {metric.value()}")

    # 同じlogitsで再更新（安定）
    metric.update(logits, 1)
    print(f"   安定後スコア: {metric.value()}")

    # ProbabilisticGTSのテスト
    try:
        print("📈 ProbabilisticGTS テスト...")
        p_metric = ProbabilisticGTS()
        p_metric.update(logits, 0)
        p_metric.update(logits, 1)
        print(f"   ProbabilisticGTS スコア: {p_metric.value()}")
        print("✅ scipy利用可能")
    except ImportError:
        print("⚠️  scipy不可用 - ProbabilisticGTSをスキップ")

    print("🎉 基本テスト完了")
