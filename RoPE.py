import torch as t
def RoPE(q_s, k_s, device='cpu'):
        '''
        unnnormalized rotary positional embedding
        '''
        ### dimensions
        batch, seqn, num_heads, d_head = q_s.shape

        ### apply elu
        q_s = elu(q_s) + 1
        k_s = elu(k_s) + 1

        ############## rotate queries and values
        d_half = d_head//2
        i_s = t.arange(1, d_half + 1)
        theta_partial = 10000**(-2*i_s/d_h)
        theta_s = theta_partial.repeat_interleave(2, dim=0)
        col_nums = t.arange(1, posn + 1)
        angles = theta_s[:,None] * col_nums[None,:]
        angles = angles.to(device)

        ### get cos and sin matrices to perform tensor product
        cos_mat = t.cos(angles)
        sin_mat = t.sin(angles)

        ### expand them
        cos_mat = cos_mat.T.unsqueeze(0).unsqueeze(2)
        sin_mat = sin_mat.T.unsqueeze(0).unsqueeze(2)

        ### Apply flipping and negation directly
        q_s_n = q_s.clone()
        k_s_n = k_s.clone()
        q_s_n[..., ::2], q_s_n[..., 1::2] = -q_s_n[..., 1::2], q_s_n[..., ::2]
        k_s_n[..., ::2], k_s_n[..., 1::2] = -k_s_n[..., 1::2], k_s_n[..., ::2]

        ### apply tensor products and sum
        q_s_rot = q_s_n * cos_mat + q_s_n * sin_mat
        k_s_rot = q=k_s_n * cos_mat + k_s_n * sin_mat

        return q_s_rot, k_s_rot