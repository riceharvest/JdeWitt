# Keep parallel tests bounded on a single-GPU workstation.
import pytest


@pytest.fixture(autouse=True)
def prohibit_gpu_initialization_in_cpu_suite(request, monkeypatch):
    if request.config.option.markexpr == 'not xpu':
        import torch
        def forbidden(*args, **kwargs):
            raise AssertionError('GPU initialization forbidden in the CPU-only suite')
        monkeypatch.setattr(torch.xpu, '_lazy_init', forbidden)
        monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)


def pytest_xdist_auto_num_workers(config):
    return 2

def pytest_collection_modifyitems(items):
    for item in items:
        if 'xpu' in item.keywords:
            item.add_marker(pytest.mark.xdist_group('b70'))
