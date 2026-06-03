# ##网格划分

MAP_SIZE = 100

def get_grid_id(x,
                y,
                grid_size):

    cell_size = MAP_SIZE / grid_size

    gx = int(x // cell_size)
    gy = int(y // cell_size)

    gx = min(gx, grid_size - 1)
    gy = min(gy, grid_size - 1)

    return gx, gy

def build_sensitivity_map(grid_size):

    sensitivity = {}

    for i in range(grid_size):
        for j in range(grid_size):

            sensitivity[(i, j)] = 0.2

    center = grid_size // 2

    sensitive_areas = [

        (center, center),

        (center, center-1),

        (center-1, center),

        (center-1, center-1)

    ]

    for area in sensitive_areas:

        sensitivity[area] = 0.9

    return sensitivity



#
# MAP_SIZE = 100
# GRID_SIZE = 10
# CELL_SIZE = MAP_SIZE / GRID_SIZE
#
# def get_grid_id(x, y):
#
#     gx = int(x // CELL_SIZE)
#     gy = int(y // CELL_SIZE)
#
#     gx = min(gx, GRID_SIZE - 1)
#     gy = min(gy, GRID_SIZE - 1)
#
#     return gx, gy
#
# def build_sensitivity_map():
#
#     sensitivity = {}
#
#     for i in range(GRID_SIZE):
#         for j in range(GRID_SIZE):
#
#             sensitivity[(i, j)] = 0.2
#
#     sensitive_areas = [
#         (4,4),
#         (4,5),
#         (5,4),
#         (5,5)
#     ]
#
#     for area in sensitive_areas:
#
#         sensitivity[area] = 0.9
#
#     return sensitivity