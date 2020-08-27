@convert_args_to_tensor([0], ['labels'])
def torch_one_hot(labels, one_hot_size):
    one_hot = torch.zeros(labels.shape[0], one_hot_size, device=labels.device)
    one_hot[torch.arange(labels.shape[0], device=labels.device), labels] = 1
    return one_hot

@convert_args_to_tensor()
def gather_nd(params, indices):
    """params is of "n" dimensions and has size [x1, x2, x3, ..., xn], indices is of 2 dimensions  and has size [num_samples, m] (m <= n)"""
    assert type(indices) == torch.Tensor
    return params[indices.transpose(0,1).long().numpy().tolist()]