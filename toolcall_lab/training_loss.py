"""Assistant-only objective with an exclusive sequence-length endpoint."""


def completion_mask(positions, offsets, lengths):
    # Position i predicts token i. A sequence of length N ends at token N-1.
    return (positions >= offsets) & (positions < lengths)


def assistant_loss(model, batch, lengths):
    import mlx.core as mx
    import mlx.nn as nn
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    positions = mx.arange(1, targets.shape[1] + 1)
    mask = completion_mask(positions, lengths[:, 0:1], lengths[:, 1:])
    ntokens = mask.sum()
    losses = nn.losses.cross_entropy(logits, targets) * mask
    return losses.astype(mx.float32).sum() / ntokens, ntokens
