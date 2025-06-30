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
GTS制御サンプラーモジュール

Generative Trajectory Stability (GTS) に基づく制御サンプリング手法を提供します。
"""

try:
    from .gts_controlled_sampler import generate_with_gts_controlled_sampling
except (ImportError, ValueError):
    # 直接実行される場合の絶対インポート
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gts_controlled_sampler import generate_with_gts_controlled_sampling

__all__ = ['generate_with_gts_controlled_sampling']
