"""Instance-level loss terms of the 3ASC 3.0 (MIL-GFE) objective."""

import torch


def focal_loss(
    prob: torch.Tensor,
    true: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    """Focal loss over the instances of a bag.

    Down-weights the easy negatives that dominate a bag of a few hundred
    variants with a single causal one.

    Args:
        prob (torch.Tensor): Predicted instance probabilities.
        true (torch.Tensor): Instance labels.
        gamma (float): Focusing parameter; larger values suppress easy examples
            more strongly.
        alpha (float): Weight of the positive class.

    Returns:
        torch.Tensor: Mean focal loss.

    See Also:
        Lin et al., Focal loss for dense object detection,
        https://arxiv.org/abs/1708.02002 (equation 5).
    """
    p_t = prob * true + (1 - prob) * (1 - true)
    modulating_factor = (1 - p_t) ** gamma
    ce_loss = torch.nn.functional.binary_cross_entropy(prob, true, reduction="none")
    alpha_t = alpha * true + (1 - alpha) * (1 - true)

    return (alpha_t * modulating_factor * ce_loss).mean()


def pointwise_ranknet_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, sigma: float
) -> torch.Tensor:
    """RankNet loss over the causal variants of a bag, as published.

    A bag has one (rarely two) causal variant, so the full pairwise RankNet
    matrix is redundant: only the rows comparing a causal variant against the
    rest carry a non-trivial target.

    The loss falls as the causal variant is ranked further above the rest of its
    bag: fed the instance probabilities, it runs from about 0.50 - the causal
    variant scored below every other variant - down to about 0.00, a full 1.0
    above them.

    Two details are kept exactly as the published model was trained; either one
    changes the optimisation trajectory, so neither has been tidied up:

    - `torch.nonzero` returns (i, j) index pairs, and indexing with that
      (n_pairs, 2) tensor selects whole *rows* i and j rather than the single
      (i, j) entries, so the mean also covers the non-causal row j and its
      targets of 0.0 and 0.5.
    - the cost is returned as `1 - c` rather than `c`, which is what makes
      minimising it push the causal variant up rather than down.

    Args:
        y_pred (torch.Tensor): Predicted instance probabilities.
        y_true (torch.Tensor): Instance labels.
        sigma (float): Scale of the pairwise sigmoid.

    Returns:
        torch.Tensor: Rank loss; lower is a better ranking.

    See Also:
        https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf

    Example:
        >>> y_pred = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> pointwise_ranknet_loss(y_pred, y_true, sigma=1.0)
        tensor(0.1625)
    """
    # Predicted probability that i outranks j, for every pair.
    diff_matrix = y_pred.view(-1, 1) - y_pred.view(1, -1)
    pij = torch.sigmoid(-sigma * diff_matrix)

    # Target: 1 where i is causal and j is not, 0.5 for ties, 0 otherwise.
    label_diff = y_true.view(-1, 1) - y_true.view(1, -1)
    input_device = y_pred.device
    pbar = torch.where(
        label_diff > 0,
        torch.tensor(1.0).to(input_device),
        torch.where(
            label_diff == 0,
            torch.tensor(0.5).to(input_device),
            torch.tensor(0.0).to(input_device),
        ),
    )

    # Rows of the pairs where i is causal and j is not - see the docstring: this
    # selects the whole row i and the whole row j, which is how it was published.
    indices = torch.nonzero(pbar == 1.0).squeeze()
    pbar = pbar[indices]
    pij = pij[indices]

    c = -pbar * torch.log(pij) - (1 - pbar) * torch.log(1 - pij)

    return 1 - c.mean()
