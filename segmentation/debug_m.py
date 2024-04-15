import math


def find_best_match_resolution(image_size, num_ps):
    patch_size0 = math.ceil(image_size[0] / num_ps[0])
    best_x = num_ps[0] * patch_size0

    # 计算最接近且不小于n的14的倍数
    patch_size1 = math.ceil(image_size[1] / num_ps[1])
    best_y = num_ps[1] * patch_size1

    return [patch_size0, patch_size1], [best_x, best_y]


#
# # 测试函数
# print(find_best_match_resolution((720, 960), (16, 16)))  # 示例输入

import math


def find_optimal_patch_and_size(image_size, depth, min_patches=9, max_patches=16):
    """
    根据输入的图像尺寸、深度和patch数量范围，找到最优的 patch size 和调整后的 image size。

    参数:
    image_size (tuple): 输入的图像宽高 (width, height)
    depth (int): 深度，patch size 必须是 2^depth 的整数倍
    min_patches (int): 分割后的最小块数
    max_patches (int): 分割后的最大块数

    返回:
    tuple: 最优的 patch size 和调整后的 image size
    """
    base = 2 ** depth
    best_patch_sizes = []
    best_adjusted_sizes = []
    best_num_patches = []

    for dimension in image_size:
        best_patch_diff = float('inf')
        best_patch_size = None
        best_adjusted_size = None
        best_patch_count = None

        # 遍历可能的patch size
        for patch_size in range(base, dimension + 1, base):
            num_patches = math.ceil(dimension / patch_size)
            if min_patches <= num_patches <= max_patches:
                adjusted_size = patch_size * num_patches
                patch_diff = abs(dimension - adjusted_size)

                # 寻找变化最小的调整后尺寸
                if patch_diff < best_patch_diff:
                    best_patch_diff = patch_diff
                    best_patch_size = patch_size
                    best_adjusted_size = adjusted_size
                    best_patch_count = num_patches

        best_patch_sizes.append(best_patch_size)
        best_adjusted_sizes.append(best_adjusted_size)
        best_num_patches.append(best_patch_count)

    return best_patch_sizes, best_adjusted_sizes, best_num_patches


# 示例使用
# # image_size = (720, 960)
# image_size = (224, 224)
# depth = 3
# patch_sizes, new_sizes, num_patches = find_optimal_patch_and_size(image_size, depth)
# print("Patch sizes:", patch_sizes)
# print("Adjusted image sizes:", new_sizes)
# print("Number of patches:", num_patches)

for k in range(1,6):
    image_size = (720 // k, 960 // k)
    depth = 4
    patch_sizes, new_sizes, num_patches = find_optimal_patch_and_size(image_size, depth)
    # print("IMG sizes:", image_size)
    print("IMG sizes:", image_size)
    print("Adjusted image sizes:", new_sizes)
    print("Patch sizes:", patch_sizes)
    print("Number of patches:", num_patches)
