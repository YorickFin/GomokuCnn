"""验证新网络结构 + 旧 checkpoint 兼容性。"""
import torch
from src.net import ValueCNN


def test_new_structure():
    model = ValueCNN()
    total = sum(p.numel() for p in model.parameters())
    print(f"新结构: hidden={model.hidden_channels}, blocks={model.num_blocks}")
    print(f"  value_conv: {model.value_conv.weight.shape}")
    print(f"  value_fc1: {model.value_fc1.weight.shape}")
    print(f"  policy_conv1: {model.policy_conv1.weight.shape}")
    print(f"  policy_conv2: {model.policy_conv2.weight.shape}")
    print(f"  总参数量: {total:,}")

    x = torch.randn(2, 3, 15, 15)
    value, logits = model(x)
    print(f"  value shape: {value.shape}, policy: {logits.shape}")
    assert value.shape == (2, 1)
    assert logits.shape == (2, 225)
    print("新结构前向传播 OK")


def test_old_checkpoint_compat():
    # 模拟旧代码结构：policy_mid=hidden//2, value_conv=1, value_dim=128
    old_model = ValueCNN(hidden_channels=64, num_blocks=2, value_dim=128,
                         value_conv_channels=1, policy_mid_channels=32)
    old_sd = old_model.state_dict()
    print(f"\n旧结构 policy_conv1: {old_sd['policy_conv1.weight'].shape}")
    print(f"旧结构 value_conv: {old_sd['value_conv.weight'].shape}")
    print(f"旧结构 value_fc1: {old_sd['value_fc1.weight'].shape}")

    restored = ValueCNN.from_state_dict(old_sd)
    print(f"恢复后 policy_conv1: {restored.policy_conv1.weight.shape}")
    print(f"恢复后 value_conv: {restored.value_conv.weight.shape}")
    print(f"恢复后 value_fc1: {restored.value_fc1.weight.shape}")

    old_model.eval()
    restored.eval()
    x = torch.randn(1, 3, 15, 15)
    with torch.no_grad():
        v1, p1 = old_model(x)
        v2, p2 = restored(x)
    assert torch.allclose(v1, v2)
    assert torch.allclose(p1, p2)
    print("旧结构 checkpoint 兼容性 OK")


def test_new_checkpoint_roundtrip():
    model = ValueCNN(hidden_channels=64, num_blocks=2)
    sd = model.state_dict()
    restored = ValueCNN.from_state_dict(sd)
    model.eval()
    restored.eval()
    x = torch.randn(1, 3, 15, 15)
    with torch.no_grad():
        v1, p1 = model(x)
        v2, p2 = restored(x)
    assert torch.allclose(v1, v2)
    assert torch.allclose(p1, p2)
    print("\n新结构 checkpoint 往返 OK")


if __name__ == "__main__":
    test_new_structure()
    test_old_checkpoint_compat()
    test_new_checkpoint_roundtrip()
    print("\n全部验证通过。")
