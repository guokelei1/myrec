import torch

from train.losses import masked_listwise_loss


def test_empty_positive_batch_returns_differentiable_zero() -> None:
    scores = torch.randn(2, 3, requires_grad=True)
    labels = torch.zeros(2, 3)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    loss = masked_listwise_loss(scores, labels, mask)
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert scores.grad is not None
    assert torch.count_nonzero(scores.grad) == 0
