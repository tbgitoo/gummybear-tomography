import torch

def get_device():
    """Select the best available PyTorch device for this host.

    Preference order: CUDA, Apple MPS, then CPU. Uses the default accelerator
    index for each backend (no explicit ``cuda:N`` selection).

    Returns:
        torch.device: ``cuda``, ``mps``, or ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")