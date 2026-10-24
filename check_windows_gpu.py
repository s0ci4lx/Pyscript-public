"""Fail clearly instead of silently running a Windows CUDA install on CPU."""
import sys

def main():
    import torch
    import torchvision
    from torchvision.ops import nms
    print("[GPU] PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    if not torch.cuda.is_available():
        print("[GPU] NVIDIA CUDA is unavailable. Update the NVIDIA driver, then rerun install_windows.bat.")
        return 1
    device = torch.device("cuda:0")
    print("[GPU]", torch.cuda.get_device_name(0))
    sample = torch.ones((32, 32), device=device)
    assert (sample @ sample).sum().item() == 32768
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
    nms(boxes, torch.tensor([.9, .8], device=device), .5)
    torch.cuda.synchronize()
    print("[GPU] CUDA tensor and torchvision detection operations passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
