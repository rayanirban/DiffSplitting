import torch

def get_n2v_input_pixel_locations(n2v_kernel_size, H,W, numpix):
    h_idx = torch.randint(0, H, (numpix,))
    w_idx = torch.randint(0, W, (numpix,))
    h_shift = torch.randint(-n2v_kernel_size//2, n2v_kernel_size//2+1, (numpix,))
    w_shift = torch.randint(-n2v_kernel_size//2, n2v_kernel_size//2+1, (numpix,))
    h_val_idx = h_idx + h_shift
    w_val_idx = w_idx + w_shift
    h_val_idx[h_val_idx < 0] = 0
    w_val_idx[w_val_idx < 0] = 0
    h_val_idx[h_val_idx >= H] = H-1
    w_val_idx[w_val_idx >= W] = W-1
    return h_idx, w_idx, h_val_idx, w_val_idx

def update_input_for_n2v(data_dict, n2v_kernel_size, n2v_p):
    data_dict['input'] = data_dict['input'].clone()
    B,C,H,W = data_dict['input'].shape
    numpix = int(data_dict['input'][0].numel() * n2v_p) * B
    # print('Numpix:', numpix)
    assert C == 1, "N2V is only implemented for single channel images."
    
    # compute the batch index
    batch_idx = torch.zeros((numpix,), dtype=torch.long).to(data_dict['input'].device)
    b_sz = numpix//B
    for i in range(B):
        batch_idx[i*b_sz: (i+1)*b_sz] = i

    
    h_idx, w_idx, h_val_idx, w_val_idx = get_n2v_input_pixel_locations(n2v_kernel_size,H,W, numpix)
    data_dict['input'][batch_idx,0,h_idx,w_idx] = data_dict['input'][batch_idx,0,h_val_idx,w_val_idx]
    return batch_idx, h_idx, w_idx, h_val_idx, w_val_idx