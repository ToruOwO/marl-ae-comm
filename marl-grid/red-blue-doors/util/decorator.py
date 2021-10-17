import torch


def within_cuda_device(f):
    """
    decorator used to wrap class methods. all variables spawned within the
    decorator will have the specified gpu id
    """

    def _wrapper(*args):
        with torch.cuda.device(args[0].gpu_id):
            return f(*args)

    return _wrapper

