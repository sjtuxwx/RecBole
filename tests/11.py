import torch

def unique_with_order(x):
    """
    输入: x (1D tensor)
    输出: 
        unique_values: 保持原顺序的唯一值
        first_indices: 这些值第一次出现的索引
    """
    # 1. 获取唯一值(自动排序)和逆向索引
    # sorted=True 是默认的，返回的 sorted_uniques 是按数值大小排序的 (如 [1, 2, 3])
    # inverse_indices 是原始 x 中的元素对应 sorted_uniques 的下标
    sorted_uniques, inverse_indices = torch.unique(x, sorted=True, return_inverse=True)
    
    # 2. 准备一个容器来存储每个唯一值第一次出现的索引
    # 初始化为由 x 的长度填充的一个大数（为了后面取最小值做准备）
    # 长度等于唯一值的数量
    first_indices_placeholder = torch.zeros(
        sorted_uniques.size(0), 
        dtype=torch.long, 
        device=x.device
    ).fill_(x.size(0))
    
    # 3. 生成原始位置的索引序列 [0, 1, 2, ...]
    perm = torch.arange(x.size(0), dtype=torch.long, device=x.device)
    
    # 4. 使用 scatter_reduce 找到每个唯一值对应的最小索引
    # 逻辑解释：
    # inverse_indices 告诉我们 x 的第 i 个元素属于哪个“唯一值类别”。
    # 我们把 perm (即原始索引) 按照 inverse_indices 撒(scatter)到 placeholder 里。
    # reduce="amin" 表示：如果同一个位置被撒入多个索引，保留最小的那个（即第一次出现的那个）。
    # 注意：scatter_reduce_ 需要 PyTorch 1.12+ 版本
    first_indices_placeholder.scatter_reduce_(
        0, 
        inverse_indices, 
        perm, 
        reduce="amin", 
        include_self=False
    )
    
    # 此时，first_indices_placeholder 里存的是 sorted_uniques 对应的第一次出现索引。
    # 但因为 sorted_uniques 是按值排序的，所以这里的索引顺序是乱的（不是按出现顺序）。
    
    # 5. 对索引进行排序，恢复“按出现顺序”排列
    # sort_indices 是排序后的索引，indices_order 是原本的排序位置
    sorted_first_indices, sort_order = first_indices_placeholder.sort()
    
    # 6. 使用排序后的索引从原始 x 中提取值，或者重新排列 sorted_uniques
    # 推荐直接用 sorted_first_indices 从 x 取值，这样最直观
    final_unique_values = x[sorted_first_indices]
    
    return final_unique_values, sorted_first_indices

# --- 测试代码 ---
input_tensor = torch.tensor([3, 1, 2, 3, 2, 5, 1])
values, indices = unique_with_order(input_tensor)

print(f"原始 Tensor: {input_tensor}")
print(f"按序去重后: {values}")  # 应该输出 [3, 1, 2, 5]
print(f"对应的索引: {indices}") # 应该输出 [0, 1, 2, 5]