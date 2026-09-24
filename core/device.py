def get_device(preference='auto'):
    if preference == 'cpu': return 'cpu'
    try:
        import torch
        if torch.cuda.is_available(): return 'cuda'
    except Exception: pass
    return 'cpu'

def gpu_info():
    try:
        import torch
        if not torch.cuda.is_available(): return {'cuda': False, 'name': None, 'vram_gb': 0}
        p = torch.cuda.get_device_properties(0)
        return {'cuda': True, 'name': torch.cuda.get_device_name(0), 'vram_gb': round(p.total_memory/1024**3,2), 'torch_cuda': torch.version.cuda}
    except Exception as e:
        return {'cuda': False, 'name': None, 'vram_gb': 0, 'error': str(e)}
