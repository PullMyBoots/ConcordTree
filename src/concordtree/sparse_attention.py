import torch
import math

import triton
import triton.language as tl

###############################################################################
# 超参数配置 （保持BLOCKM等于BLOCKN）
BLOCK_M = 16
BLOCK_N = 16
COMPILED_ATTENTION_NUM_WARPS = 1
################################################################################
# layout_ulits


@triton.jit
def _compile_historical_pair_masks_kernel(
    active_indices,
    active_counts,
    quartet_matrix,
    pair_masks,
    stride_active_M,
    stride_active_N,
    quartet_matrix_M,
    quartet_matrix_N,
    stride_mask_M,
    stride_mask_N,
    stride_mask_R,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Capture the exact predicate seen by the historical Triton load."""

    query_block = tl.program_id(0)
    active_slot = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_species = tl.arange(0, 64)
    destination = (
        pair_masks
        + query_block * stride_mask_M
        + active_slot * stride_mask_N
        + offs_m * stride_mask_R
    )
    active_count = tl.load(active_counts + query_block)
    if active_slot < active_count:
        key_block = tl.load(
            active_indices
            + query_block * stride_active_M
            + active_slot * stride_active_N
        )
        query_rows = query_block * BLOCK_M + offs_m
        key_rows = key_block * BLOCK_N + offs_n
        query_ptrs = (
            quartet_matrix
            + query_rows[:, None] * quartet_matrix_M
            + offs_species[None, :] * quartet_matrix_N
        )
        key_ptrs = (
            quartet_matrix
            + key_rows[:, None] * quartet_matrix_M
            + offs_species[None, :] * quartet_matrix_N
        )
        overlap = tl.dot(tl.load(query_ptrs), tl.trans(tl.load(key_ptrs)))
        weights = 1 << offs_n
        row_masks = tl.sum(
            (overlap >= 3).to(tl.int32) * weights[None, :], axis=1
        )
        tl.store(destination, row_masks)
    else:
        tl.store(destination, 0)

def compress_block_layout(coeff_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted active key-block indices and row counts.

    The dense block mask is a frozen mathematical sparsity pattern.  Sorting
    the active indices keeps the historical left-to-right accumulation order.
    The width is padded to a multiple of 16 for a regular Triton loop.
    """

    if coeff_blocks.ndim != 2:
        raise ValueError("coeff_blocks must be a matrix")
    mask = coeff_blocks.detach().to(device="cpu", dtype=torch.bool)
    counts = mask.sum(dim=1, dtype=torch.int32)
    maximum = int(counts.max().item())
    width = ((maximum + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    indices = torch.full((mask.shape[0], width), -1, dtype=torch.int32)
    for row in range(mask.shape[0]):
        active = torch.nonzero(mask[row], as_tuple=False).flatten().to(torch.int32)
        indices[row, : active.numel()] = active
    return indices.to(coeff_blocks.device), counts.to(coeff_blocks.device)


def define_historical_quartet_tail(quartet_matrix: torch.Tensor) -> torch.Tensor:
    """Return the same tensor view with a defined zero tail in its storage.

    The historical kernel reads 64 values from each 48-value row.  Real rows
    therefore intentionally retain the frozen cross-row aliasing, while the
    final padded row receives 16 in-allocation zeros instead of reading an
    unrelated CUDA allocation.  Shape and strides remain unchanged.
    """

    if quartet_matrix.ndim != 2 or not quartet_matrix.is_contiguous():
        raise ValueError("quartet_matrix must be a contiguous matrix")
    storage = torch.cat(
        [quartet_matrix.view(-1), quartet_matrix.new_zeros(16)]
    )
    return torch.as_strided(
        storage, size=quartet_matrix.shape, stride=quartet_matrix.stride()
    )


def compile_historical_pair_masks(
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    quartet_matrix: torch.Tensor,
    *,
    chunk_blocks: int = 1024,
) -> torch.Tensor:
    """Compile the frozen token predicate into packed 16-bit row masks.

    The historical kernel presents the 48-column quartet tensor to Triton as
    a 64-column strided view.  For real tokens this makes columns 48--63 alias
    the first 16 columns of the following row.  The checkpoint was evaluated
    with that effective relation, so the compatibility backend records it
    explicitly rather than silently changing the graph.

    The result has shape ``(query_blocks, active_slots, 16)``.  Each int32
    stores the 16 key-token predicates for one query token in one active tile.
    Appended zeros give the padded tail a deterministic definition; real-token
    rows exactly retain the historical in-bounds 64-wide/48-stride predicate.
    """

    if active_indices.ndim != 2:
        raise ValueError("active_indices must be a matrix")
    if active_counts.shape != (active_indices.shape[0],):
        raise ValueError("active_counts must have one entry per query block")
    if quartet_matrix.ndim != 2 or quartet_matrix.shape[0] != active_indices.shape[0] * BLOCK_M:
        raise ValueError("quartet matrix does not match the active block layout")
    if quartet_matrix.stride(1) != 1 or quartet_matrix.stride(0) != quartet_matrix.shape[1]:
        raise ValueError("historical pair-mask compilation requires a contiguous row-major quartet matrix")
    if quartet_matrix.shape[1] > 64:
        raise ValueError("quartet matrix is wider than the historical 64-column view")
    if chunk_blocks < 1:
        raise ValueError("chunk_blocks must be positive")

    device = quartet_matrix.device
    query_blocks = torch.arange(active_indices.shape[0], device=device)
    active_slots = torch.arange(active_indices.shape[1], device=device)
    valid_slots = active_slots.unsqueeze(0) < active_counts.unsqueeze(1)
    query_block, active_slot = torch.nonzero(valid_slots, as_tuple=True)
    key_block = active_indices[query_block, active_slot].to(torch.long)

    # Make the historical strided over-read explicit and defined at the tail.
    storage = torch.cat(
        [
            quartet_matrix.contiguous().view(-1),
            quartet_matrix.new_zeros(64),
        ]
    )
    effective_quartets = torch.as_strided(
        storage,
        size=(quartet_matrix.shape[0], 64),
        stride=(quartet_matrix.stride(0), quartet_matrix.stride(1)),
    )
    row_offsets = torch.arange(BLOCK_M, device=device)
    bit_weights = (1 << torch.arange(BLOCK_N, device=device, dtype=torch.int32))
    packed = torch.zeros(
        (*active_indices.shape, BLOCK_M), dtype=torch.int32, device=device
    )
    if quartet_matrix.is_cuda:
        grid = (active_indices.shape[0], active_indices.shape[1])
        _compile_historical_pair_masks_kernel[grid](
            active_indices,
            active_counts,
            quartet_matrix,
            packed,
            active_indices.stride(0),
            active_indices.stride(1),
            quartet_matrix.stride(0),
            quartet_matrix.stride(1),
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            BLOCK_M,
            BLOCK_N,
        )
        return packed

    for start in range(0, query_block.numel(), chunk_blocks):
        stop = min(start + chunk_blocks, query_block.numel())
        q_rows = query_block[start:stop, None] * BLOCK_M + row_offsets[None, :]
        k_rows = key_block[start:stop, None] * BLOCK_N + row_offsets[None, :]
        overlap = torch.bmm(
            effective_quartets[q_rows],
            effective_quartets[k_rows].transpose(1, 2),
        ) >= 3
        row_masks = torch.sum(
            overlap.to(torch.int32) * bit_weights[None, None, :],
            dim=2,
            dtype=torch.int32,
        )
        packed[query_block[start:stop], active_slot[start:stop]] = row_masks
    return packed

def _get_layout_quartetmatrix(sample_quartets, device="cuda", block_size=BLOCK_M):

    def calculate_shared_species_mask(positionidx, species_threshold, pad=None):
        """
        计算共享物种数目的掩码，适用于每个物种占用多个编码位的情况。

        :param positionidx: torch.Tensor, shape (quadruplet_num, 128), dtype=torch.bool
        :param species_num: int, 物种总数
        :param species_threshold: int, 最小共享物种数目
        :param pad: int, 如果不为 None，则将 mask 的行列填充到 pad 的倍数
        :return: torch.Tensor, 注意力掩码，形状为 (quadruplet_num, quadruplet_num)，如果 pad 不为 None，则形状为 (padded_quadruplet_num, padded_quadruplet_num)
        """

        shared_species = torch.matmul(
            positionidx.to(torch.float16), positionidx.transpose(0, 1).to(torch.float16)
        )
        # 使用布尔类型掩码
        mask = shared_species >= species_threshold

        # 如果需要填充
        if pad is not None:
            current_size = mask.size(0)
            if current_size % pad != 0:
                padded_size = ((current_size // pad) + 1) * pad
                padded_mask = torch.zeros((padded_size, padded_size), device=mask.device, dtype=torch.bool)
                padded_mask[:current_size, :current_size] = mask
                return padded_mask

        return mask


    species_threshold = 3
    
    # Load the species data
    
    data = sample_quartets
    species_mapping = {f'Sp{i}': i for i in range(48)}

    # Construct quartet matrix
    quartetmatrix = torch.zeros((len(data), 48), device=device)
    for row, group in enumerate(data):
        for species in group:
            if species in species_mapping:
                idx = species_mapping[species]
                start = idx
                quartetmatrix[row, start:start + 1] = 1

    # Calculate the shared species mask
    mask = calculate_shared_species_mask(quartetmatrix, species_threshold, block_size)

    # Pad the quartetmatrix to match the padded mask size
    padded_size = mask.size(0)
    if quartetmatrix.size(0) < padded_size:
        padded_quartetmatrix = torch.zeros((padded_size, quartetmatrix.size(1)), device=device)
        padded_quartetmatrix[:quartetmatrix.size(0), :] = quartetmatrix
    else:
        padded_quartetmatrix = quartetmatrix

    return mask, padded_quartetmatrix

def _partition_bool_matrix(input_matrix, M=BLOCK_M, H=BLOCK_M, device='cuda'):
    """
    使用PyTorch高效地将输入布尔矩阵按照M行和H列进行分块，
    如果一个块中有任何True，则在输出矩阵中对应的位置设为True，否则为False。

    参数:
    input_matrix (torch.Tensor): 输入的a x a布尔矩阵
    M (int): 分块的行跨度
    H (int): 分块的列跨度
    device (str): 设备，默认使用'cuda'（GPU），如果没有GPU可设为'cpu'

    返回:
    torch.Tensor: 分块后的(a/M) x (a/H)布尔矩阵
    """
    # 将输入矩阵移动到指定设备
    input_matrix = input_matrix.to(device)
    
    a, b = input_matrix.shape
    assert a % M == 0 and b % H == 0, "矩阵的行数和列数必须能被M和H整除"
    
    # 使用张量视图操作将矩阵分块
    reshaped_matrix = input_matrix.view(a // M, M, b // H, H)

    # 在每个块中检查是否存在 True
    # 分两步调用 any() 以兼容旧版本 PyTorch
    block_results = reshaped_matrix.any(dim=1).any(dim=2)

    return block_results

def get_SparseLayout_SpeciesEncoding(sample_quartets, species_num):
        layout, quartet_matrix = _get_layout_quartetmatrix(sample_quartets)         
        coeff_blocks = _partition_bool_matrix(layout)         
        del layout         
        torch.cuda.empty_cache()

        species_mapping = {f'Sp{i}': i for i in range(species_num)}
        species_encoding = torch.zeros((len(sample_quartets), species_num), dtype=torch.float32)
        for row, group in enumerate(sample_quartets):
            for species in group:
                if species in species_mapping:
                    idx = species_mapping[species]  
                    species_encoding[row, idx] = 1

        return coeff_blocks, quartet_matrix, species_encoding

################################################################################
@triton.jit     
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              coeff_blocks, quartet_matrix,

              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              stride_coeff_M, stride_coeff_N,
              quartet_matrix_M, quartet_matrix_N,
              stride_Mz, stride_My, stride_Ml,

              Z, H, N_CTX, HEAD_DIM: tl.constexpr,                       
              BLOCK_M: tl.constexpr,  #                         
              BLOCK_N: tl.constexpr,  #                                                    
              ):
    
    pid = tl.program_id(0)       # 计算[start_m, start_m+BLOCK_M]
    off_hz = tl.program_id(1)        # 由于head计算的独立性，可以把bz*headnum统一看成bz，off_hz对应计算第off_hz个bz
    off_z = off_hz // H              # 具体的batch_No
    off_h = off_hz % H               # 具体的head_No
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh #获取起始位置，也就是具体（batch_No，head_No）对应的矩阵
    coeff_blocks_offset = pid * stride_coeff_M
    M_offset = off_z.to(tl.int64) * stride_Mz + off_h.to(tl.int64) * stride_My

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    coeff_ptr = coeff_blocks + coeff_blocks_offset
    q_quartet_matrix_ptr = tl.make_block_ptr(
        base=quartet_matrix,
        shape=(N_CTX, 64),
        strides=(quartet_matrix_M, quartet_matrix_N),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, 64),
        order=(1, 0)
    )
    k_quartet_matrix_ptr = tl.make_block_ptr(
        base=quartet_matrix,
        shape=(N_CTX, 64),
        strides=(quartet_matrix_M, quartet_matrix_N),
        offsets=(0, 0),
        block_shape=(BLOCK_N, 64),
        order=(1, 0)
    )
      
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 999
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
  
    # forward
    q = tl.load(Q_block_ptr)
    q_quartet_matrix = tl.load(q_quartet_matrix_ptr)
    
    for start_n in range(0, N_CTX, BLOCK_N):
        non_zero_num = tl.load(coeff_ptr)
        coeff_ptr = coeff_ptr + 1
        
        if non_zero_num:
            tl.debug_barrier()
            # --- compute qk ----
            k = tl.load(K_block_ptr)
            qk = tl.dot(q, tl.trans(k))
            k_quartet_matrix = tl.load(k_quartet_matrix_ptr)
            qk_mask = tl.dot(q_quartet_matrix, tl.trans(k_quartet_matrix))
            qk_mask = tl.where(qk_mask >= 3, 0, -float("inf"))
            qk = qk + qk_mask

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp(qk)
            l_ij = tl.sum(p, 1)
            
            # -- update m_i and l_i
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            # -- update output accumulator --
            acc = acc * alpha[:, None]
            
            # update acc
            v = tl.load(V_block_ptr)
            
           
            acc += tl.dot(p, v)
            m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        k_quartet_matrix_ptr = tl.advance(k_quartet_matrix_ptr, (BLOCK_N, 0))

    # epilogue
    m_i += tl.math.log(l_i)
    acc = acc / l_i[:, None]

    m_block_ptrs = tl.make_block_ptr(
        base=M + M_offset,
        shape=(N_CTX),
        strides=(1, ),
        offsets=(pid * BLOCK_M ),
        block_shape=(BLOCK_M, ),
        order=(0, )
    )
    tl.store(m_block_ptrs, m_i)
    tl.store(O_block_ptr, acc)

#################################################################################

@triton.jit
def _attn_fwd_indexed(
              Q, K, V, sm_scale, Out,
              active_indices, active_counts, quartet_matrix,

              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              stride_active_M, stride_active_N,
              quartet_matrix_M, quartet_matrix_N,

              Z, H, N_CTX, HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              ):
    """Inference traversal over sorted nonzero blocks only."""

    pid = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = (
        off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    )

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_species = tl.arange(0, 64)
    query_rows = pid * BLOCK_M + offs_m

    q_ptrs = (
        Q + qvk_offset + query_rows[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q = tl.load(q_ptrs)
    q_quartet_ptrs = (
        quartet_matrix + query_rows[:, None] * quartet_matrix_M
        + offs_species[None, :] * quartet_matrix_N
    )
    q_quartet_matrix = tl.load(q_quartet_ptrs)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 999
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    active_count = tl.load(active_counts + pid)

    # active_count is uniform within a program, so this loop has no divergent
    # per-thread branch and requires no explicit debug barrier.
    for active_offset in tl.range(0, active_count):
        key_block = tl.load(
            active_indices
            + pid * stride_active_M
            + active_offset * stride_active_N
        )
        key_rows = key_block * BLOCK_N + offs_n
        k_ptrs = (
            K + qvk_offset + key_rows[:, None] * stride_kn
            + offs_d[None, :] * stride_kk
        )
        v_ptrs = (
            V + qvk_offset + key_rows[:, None] * stride_vk
            + offs_d[None, :] * stride_vn
        )
        k_quartet_ptrs = (
            quartet_matrix + key_rows[:, None] * quartet_matrix_M
            + offs_species[None, :] * quartet_matrix_N
        )

        k = tl.load(k_ptrs)
        qk = tl.dot(q, tl.trans(k))
        k_quartet_matrix = tl.load(k_quartet_ptrs)
        qk_mask = tl.dot(q_quartet_matrix, tl.trans(k_quartet_matrix))
        qk_mask = tl.where(qk_mask >= 3, 0, -float("inf"))
        qk = qk + qk_mask

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * sm_scale)
        qk = qk * sm_scale - m_ij[:, None]
        p = tl.math.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs)
        # Triton requires both dot operands to share a dtype.  This is a
        # no-op for the frozen FP32 path and permits explicit BF16/FP16
        # inference probes without changing the FP32 accumulation buffer.
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = (
        Out + qvk_offset + query_rows[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    tl.store(o_ptrs, acc)


@triton.jit
def _attn_fwd_compiled_mask(
              Q, K, V, sm_scale, Out,
              active_indices, active_counts, pair_masks,

              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              stride_active_M, stride_active_N,
              stride_mask_M, stride_mask_N, stride_mask_R,

              Z, H, N_CTX, HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              ):
    """Inference attention over a precompiled static token relation."""

    pid = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = (
        off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    )

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    query_rows = pid * BLOCK_M + offs_m

    q_ptrs = (
        Q + qvk_offset + query_rows[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q = tl.load(q_ptrs)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 999
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    active_count = tl.load(active_counts + pid)

    for active_offset in tl.range(0, active_count):
        key_block = tl.load(
            active_indices
            + pid * stride_active_M
            + active_offset * stride_active_N
        )
        key_rows = key_block * BLOCK_N + offs_n
        k_ptrs = (
            K + qvk_offset + key_rows[:, None] * stride_kn
            + offs_d[None, :] * stride_kk
        )
        v_ptrs = (
            V + qvk_offset + key_rows[:, None] * stride_vk
            + offs_d[None, :] * stride_vn
        )
        row_masks = tl.load(
            pair_masks
            + pid * stride_mask_M
            + active_offset * stride_mask_N
            + offs_m * stride_mask_R
        )
        allowed = ((row_masks[:, None] >> offs_n[None, :]) & 1) != 0

        k = tl.load(k_ptrs)
        qk = tl.dot(q, tl.trans(k))
        qk_mask = tl.where(allowed, 0.0, -float("inf"))
        qk = qk + qk_mask

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * sm_scale)
        qk = qk * sm_scale - m_ij[:, None]
        p = tl.math.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs)
        # Keep FP32 accumulation while allowing reduced-precision Q/K/V.
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = (
        Out + qvk_offset + query_rows[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    tl.store(o_ptrs, acc)


def indexed_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    quartet_matrix: torch.Tensor,
) -> torch.Tensor:
    """Launch the inference-only indexed sparse-attention kernel."""

    if torch.is_grad_enabled():
        raise RuntimeError("indexed sparse attention is inference-only")
    batch_size, head_num, n_ctx, head_dim = q.shape
    if active_indices.shape[0] != n_ctx // BLOCK_M:
        raise ValueError("active block layout does not match sequence length")
    output = torch.empty_like(q)
    grid = (n_ctx // BLOCK_M, batch_size * head_num)
    _attn_fwd_indexed[grid](
        q, k, v, 1.0 / math.sqrt(head_dim), output,
        active_indices, active_counts, quartet_matrix,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        active_indices.stride(0), active_indices.stride(1),
        quartet_matrix.stride(0), quartet_matrix.stride(1),
        batch_size, head_num, n_ctx, head_dim,
        BLOCK_M, BLOCK_N,
    )
    return output


def compiled_mask_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_indices: torch.Tensor,
    active_counts: torch.Tensor,
    pair_masks: torch.Tensor,
) -> torch.Tensor:
    """Launch inference attention with the static pair predicate precompiled."""

    if torch.is_grad_enabled():
        raise RuntimeError("compiled-mask sparse attention is inference-only")
    batch_size, head_num, n_ctx, head_dim = q.shape
    if active_indices.shape[0] != n_ctx // BLOCK_M:
        raise ValueError("active block layout does not match sequence length")
    expected = (*active_indices.shape, BLOCK_M)
    if tuple(pair_masks.shape) != expected or pair_masks.dtype != torch.int32:
        raise ValueError(f"pair_masks must have shape {expected} and dtype int32")
    output = torch.empty_like(q)
    grid = (n_ctx // BLOCK_M, batch_size * head_num)
    _attn_fwd_compiled_mask[grid](
        q, k, v, 1.0 / math.sqrt(head_dim), output,
        active_indices, active_counts, pair_masks,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        active_indices.stride(0), active_indices.stride(1),
        pair_masks.stride(0), pair_masks.stride(1), pair_masks.stride(2),
        batch_size, head_num, n_ctx, head_dim,
        BLOCK_M, BLOCK_N,
        num_warps=COMPILED_ATTENTION_NUM_WARPS,
    )
    return output

#################################################################################

@triton.jit
def _attn_bwd_preprocess(O, DO, Delta,
                         
                         stride_o_bz, stride_o_hd, stride_o_seq, stride_o_hdim,
                         stride_do_bz, stride_do_hd, stride_do_seq, stride_do_hdim,
                         stride_M_bz, stride_M_hd, stride_M_seq,

                         BATCH, N_HEAD, N_CTX,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    
    off_bz = tl.program_id(1)
    off_h = tl.program_id(2)
    off_m = tl.program_id(0)
    # make_ptr
    DO_ptr = DO + (off_bz * stride_do_bz) + (off_h * stride_do_hd)
    do_block_ptr = tl.make_block_ptr(
        base=DO_ptr,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_do_seq, stride_do_hdim),
        offsets=(off_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1,0)
    )

    O_ptr = O + (off_bz * stride_o_bz) + (off_h * stride_o_hd)
    o_block_ptr = tl.make_block_ptr(
        base=O_ptr,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_o_seq, stride_o_hdim),
        offsets=(off_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1,0)
    )

    Delta_ptr = Delta + (off_bz * stride_M_bz) + (off_h * stride_M_hd)
    delta_block_ptr = tl.make_block_ptr(
        base=Delta_ptr,
        shape=(N_CTX, ),
        strides=(stride_M_seq, ),
        offsets=(off_m * BLOCK_M, ),
        block_shape=(BLOCK_M, ),
        order=(0, )
    )
    # compute
    
    o = tl.load(o_block_ptr)
    do = tl.load(do_block_ptr)
    delta = tl.sum(o * do, axis=1)
    
    # write-back
    tl.store(delta_block_ptr, delta)

@triton.jit
def _attn_bwd(
                Q, K, V, sm_scale,
                DO, DQ, DK, DV,
                M, D,
                coeff_blocks, quartet_matrix,
                
                stride_z, stride_h, stride_tok, stride_d,
                stride_dz, stride_dh, stride_dok, stride_dd,
                stride_deltaz, stride_deltay, stride_deltax,
                coeff_blocks_y, coeff_blocks_x,
                quartet_matrix_stride_y, quartet_matrix_stride_x,
                
                H, N_CTX,  #
                BLOCK_M: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                HEAD_DIM: tl.constexpr,
            ):
    
    off_seq = tl.program_id(0)
    off_bz = tl.program_id(1)
    off_hd = tl.program_id(2)

    # offset pointers for batch/head
    Q += off_bz * stride_z + off_hd * stride_h
    K += off_bz * stride_z + off_hd * stride_h
    V += off_bz * stride_z + off_hd * stride_h
    DO += off_bz * stride_z + off_hd * stride_h
    DQ += off_bz * stride_dz + off_hd * stride_dh
    DK += off_bz * stride_dz + off_hd * stride_dh
    DV += off_bz * stride_dz + off_hd * stride_dh
    M += off_bz * stride_deltaz + off_hd * stride_deltay
    D += off_bz * stride_deltaz + off_hd * stride_deltay
 

    #================dk dv==================# 
    # make_block_ptr
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(off_seq * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(off_seq * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )

    dK_block_ptr = tl.make_block_ptr(
        base=DK,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(off_seq * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    dV_block_ptr = tl.make_block_ptr(
        base=DV,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(off_seq * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )

    M_block_ptr = tl.make_block_ptr(
        base = M,
        shape=(N_CTX, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(BLOCK_M, ),
        order=(0, ),
    )
    D_block_ptr = tl.make_block_ptr(
        base = D,
        shape=(N_CTX, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(BLOCK_M, ),
        order=(0, ),
    )

    coeff_ptr = coeff_blocks + off_seq
    K_quartet_matrix_ptr = tl.make_block_ptr(
        base=quartet_matrix,
        shape=(N_CTX, 64),
        strides=(quartet_matrix_stride_y, quartet_matrix_stride_x),
        offsets=(off_seq * BLOCK_N, 0),
        block_shape=(BLOCK_M, 64),
        order=(1, 0)
    )
    Q_quartet_matrix_ptr = tl.make_block_ptr(
        base=quartet_matrix,
        shape=(N_CTX, 64),
        strides=(quartet_matrix_stride_y, quartet_matrix_stride_x),
        offsets=(0, 0),
        block_shape=(BLOCK_N, 64),
        order=(1, 0)
    )

    # Compute d
    K_quartet = tl.load(K_quartet_matrix_ptr)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    dk = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for idx in range(N_CTX // BLOCK_M):

        coeff = tl.load(coeff_ptr)
        if coeff:
            tl.debug_barrier()
            # Compute P.
            q = tl.load(Q_block_ptr)
            kqT = tl.dot(k, tl.trans(q))

            Q_quartet = tl.load(Q_quartet_matrix_ptr)
            kqT_mask = tl.dot(K_quartet, tl.trans(Q_quartet))
            kqT_mask = tl.where(kqT_mask >= 3, 0, -float("inf"))

            m = tl.load(M_block_ptr)
            kqT = kqT * sm_scale - m[None, :]
            kqT = kqT + kqT_mask
            pT = tl.math.exp(kqT)

            # Compute dV.
            do = tl.load(dO_block_ptr)
            dv += tl.dot(pT, do)

            # Compute dP and dS.
            dp = tl.dot(do, tl.trans(v))
            Di = tl.load(D_block_ptr)
            ds = tl.trans(pT) * (dp - Di[:, None]) * sm_scale

            # Compute dK
            dk += tl.dot(tl.trans(ds), q)
        
        # Increment pointers.
        coeff_ptr += coeff_blocks_y
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (BLOCK_M, 0))
        M_block_ptr = tl.advance(M_block_ptr, (BLOCK_M, ))
        D_block_ptr = tl.advance(D_block_ptr, (BLOCK_M, ))
        Q_quartet_matrix_ptr = tl.advance(Q_quartet_matrix_ptr, (BLOCK_M, 0))

    tl.store(dK_block_ptr, dk)
    tl.store(dV_block_ptr, dv)
    
    #================dq====================#

    # make_block_ptr
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(off_seq * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )

    dQ_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(off_seq * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_tok, stride_d),
        offsets=(off_seq * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )

    M_block_ptr = tl.make_block_ptr(
        base = M,
        shape=(N_CTX, ),
        strides=(1, ),
        offsets=(off_seq * BLOCK_M, ),
        block_shape=(BLOCK_M, ),
        order=(0, ),
    )
    D_block_ptr = tl.make_block_ptr(
        base = D,
        shape=(N_CTX, ),
        strides=(1, ),
        offsets=(off_seq * BLOCK_M, ),
        block_shape=(BLOCK_M, ),
        order=(0, ),
    )

    coeff_ptr = coeff_blocks + off_seq * coeff_blocks_y
    K_quartet_matrix_ptr = tl.make_block_ptr(
        base=quartet_matrix,
        shape=(N_CTX, 64),
        strides=(quartet_matrix_stride_y, quartet_matrix_stride_x),
        offsets=(0, 0),
        block_shape=(BLOCK_M, 64),
        order=(1, 0)
    )
    Q_quartet_matrix_ptr = tl.make_block_ptr(
        base=quartet_matrix,
        shape=(N_CTX, 64),
        strides=(quartet_matrix_stride_y, quartet_matrix_stride_x),
        offsets=(off_seq * BLOCK_N, 0),
        block_shape=(BLOCK_N, 64),
        order=(1, 0)
    )

    # Compute
    q = tl.load(Q_block_ptr)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m = tl.load(M_block_ptr)
    d = tl.load(D_block_ptr)
    do = tl.load(dO_block_ptr)
    Q_quartet = tl.load(Q_quartet_matrix_ptr)
    
    for idx in range(N_CTX // BLOCK_M):
        coeff = tl.load(coeff_ptr)
        if coeff:
            tl.debug_barrier()
            # Compute p dp ds.
            k = tl.load(K_block_ptr)
            qkT = tl.dot(q, tl.trans(k))
            K_quartet = tl.load(K_quartet_matrix_ptr)
            qk_mask = tl.dot(Q_quartet, tl.trans(K_quartet))
            qk_mask = tl.where(qk_mask >= 3, 0, -float("inf"))
            m = tl.load(M_block_ptr)
            qkT = qkT * sm_scale - m[:, None]
            qkT = qkT + qk_mask
            p = tl.math.exp(qkT)

            v = tl.load(V_block_ptr)
            dp = tl.dot(do, tl.trans(v))

            ds = p * (dp - d[:, None]) * sm_scale
            
            # Compute dq.
            dq += tl.dot(ds, k)
        
        # Increment pointers.
        coeff_ptr += 1
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_quartet_matrix_ptr = tl.advance(K_quartet_matrix_ptr, (BLOCK_N, 0))

    tl.store(dQ_block_ptr, dq)

#####################################################################################

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, coeff_blocks, quartet_matrix):

        BATCH_SIZE, HEAD_NUM, N_CTX, HEAD_DIM = q.shape[0], q.shape[1], q.shape[2], q.shape[3]

        o = torch.empty_like(q)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        
        grid = (N_CTX // BLOCK_M, BATCH_SIZE * HEAD_NUM)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

        _attn_fwd[grid](
                q, k, v, sm_scale, M, o,
                coeff_blocks, quartet_matrix,

                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                coeff_blocks.stride(0), coeff_blocks.stride(1),
                quartet_matrix.stride(0), quartet_matrix.stride(1),
                M.stride(0), M.stride(1), M.stride(2),

                q.shape[0], q.shape[1], q.shape[2], HEAD_DIM,
                BLOCK_M, BLOCK_N
            )

        ctx.save_for_backward(q, k, v, o, M, coeff_blocks, quartet_matrix)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, coeff_blocks, quartet_matrix = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        # 这里期望的do的维度是（batchszie, headnum, seqlen, headdim）,但是实际内存中的存储是（batchsize， seqlen，headnum, headdim）
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride() 
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        #####
        BATCH, N_HEAD, N_CTX, H_DIM = q.shape[:4]
        assert N_CTX % BLOCK_M == 0
        pre_grid = (N_CTX // BLOCK_M, BATCH, N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=BLOCK_M, HEAD_DIM=ctx.HEAD_DIM  #
        )

        #####
        assert M.stride() == delta.stride()
        
        grid = (N_CTX // BLOCK_N, BATCH,  N_HEAD)
        _attn_bwd[grid](
            q, k, v, sm_scale,
            do, dq, dk, dv,
            M, delta,
            coeff_blocks, quartet_matrix,
            
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            coeff_blocks.stride(0), coeff_blocks.stride(1),
            quartet_matrix.stride(0), quartet_matrix.stride(1),
            
            N_HEAD, N_CTX,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
            HEAD_DIM=ctx.HEAD_DIM,
        )

        return dq, dk, dv, None, None, None


attention = _attention.apply

################################################################################
