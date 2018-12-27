from typing import Callable, List, Optional, Tuple, Dict, Union
import torch

def integrated_gradients(model:  Union[torch.nn.Module, Callable],
                         inputs: Dict[str, torch.Tensor],
                         ig_key: str,
                         target_label_idxs: List[int],
                         device: str,
                         steps:  Optional[int] = 50,
                         baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes integrated gradients for a given network and prediction label.
    Integrated gradients is a technique for attributing a deep network's
    prediction to its input features. It was introduced by:
    https://arxiv.org/abs/1703.01365

    This method only applies to classification networks, i.e., networks 
    that predict a probability distribution across two or more class labels.

    Access to the specific network is provided to the method via a 'model' instance or function,
    provided as argument to this method. The instance is an instance of 'torch.nn.Module', or the 
    forward function of model.

    Args:
        - model: An instance of 'torch.nn.Module', or an foward function of model.
        - inputs: The input for the given 'model', it should be an dict, and every key is 
        corresponding to the parameter of the 'forward' function of the given 'model'.
        **every input should be batched**, the tensor size should be [batch_size, ...]
        - ig_key: The specific key of input for which integrated gradients must be computed.
        - target_label_idxs:  The index of the target class for which gradients must be obtained.
        - device: Tensor device
        - steps: [optional] Number of intepolation steps between the baseline
        and the input used in the integrated gradients computation. These
        steps along determine the integral approximation error. By default,
        steps is set to 50.
        - baseline: [optional] The baseline input used in the integrated
        gradients computation. If None (default), the all zero tensor with
        the same shape as the input (i.e., 0*input) is used as the baseline.
        The provided baseline and input must have the same shape. 
        
    Returns:
        integrated_gradients: The integrated_gradients of the prediction for the
        provided prediction label to the input. It is an tensor of the same shape as that of
        the input.
    """
    scaled_inputs = {}
    assert ig_key in inputs
    for key, input in inputs.items():
        if key == ig_key:
            if baseline is None:
                baseline = 0 * input
            scaled_input = [baseline + (float(i) / steps) * (input - baseline) for i in range(steps + 1)]
        else:
            scaled_input = [input for i in range(steps + 1)]
        
        scaled_input = torch.stack(scaled_input, dim = 1)
        bs = scaled_input.size(0)
        scaled_input = scaled_input.view(bs * (steps + 1), *scaled_input.size()[2:])

        if key == ig_key:
            scaled_input = torch.tensor(scaled_input, requires_grad = True, device = device)
        else:
            scaled_input = torch.tensor(scaled_input, requires_grad = False, device = device)

        scaled_inputs[key] = scaled_input 
    # create indexes corresponding to the scaled_inputs
    target_label_idxs = torch.stack([target_label_idxs for i in range(steps + 1)], dim=1)
    target_label_idxs = target_label_idxs.view(bs * (steps + 1), 1)

    output = model(**scaled_inputs)
    output = torch.gather(output, dim=1, index=target_label_idxs).squeeze(-1)

    model.zero_grad()
    output.backward(torch.zeros(output.size(), device = device) + 1)

    gradients = scaled_inputs[ig_key].grad.detach()
    gradients = gradients.view(bs, steps + 1, *gradients.size()[1:])
    avg_grads = gradients.mean(dim=1)
    integrated_gradients = (inputs[ig_key] - baseline) * avg_grads

    return integrated_gradients

