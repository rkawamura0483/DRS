#!/usr/bin/env python3
"""
Quick test script to verify our adaptive scheduling fixes
"""

import torch
from adaptive_scheduler import AdaptiveInferenceScheduler


def test_scheduler_with_none():
    """Test scheduler with None values to reproduce the error"""
    print("Testing scheduler with None values...")

    scheduler = AdaptiveInferenceScheduler(
        min_block_size=8,
        max_block_size=32,
        base_confidence_threshold=0.8
    )

    # Test with None values (should not crash now)
    try:
        next_size, adapted_threshold, metrics = scheduler.step(
            logits=None,
            tokens=None,
            mask_index=None,
            step_num=1,
            total_steps=10
        )
        print(
            f"✅ None test passed: block_size={next_size}, threshold={adapted_threshold:.3f}")
        print(
            f"   Metrics: confidence={metrics['confidence']}, entropy={metrics['entropy']}")
    except Exception as e:
        print(f"❌ None test failed: {e}")
        return False

    # Test with valid tensors
    try:
        dummy_logits = torch.randn(1, 16, 1000)
        dummy_tokens = torch.randint(0, 1000, (1, 16))
        dummy_mask = torch.ones(1, 16, dtype=torch.bool)

        next_size, adapted_threshold, metrics = scheduler.step(
            logits=dummy_logits,
            tokens=dummy_tokens,
            mask_index=dummy_mask,
            step_num=1,
            total_steps=10
        )
        print(
            f"✅ Valid tensor test passed: block_size={next_size}, threshold={adapted_threshold:.3f}")
        print(
            f"   Metrics: confidence={metrics['confidence']:.3f}, entropy={metrics['entropy']:.3f}")
    except Exception as e:
        print(f"❌ Valid tensor test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_scheduler_with_none()
    if success:
        print("\n🎉 All tests passed! The fixes appear to be working.")
    else:
        print("\n💥 Some tests failed. Need more fixes.")
