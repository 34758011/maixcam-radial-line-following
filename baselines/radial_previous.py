from maix import camera, display, image, app
import cv2
import numpy as np
import math
# 保持原有摄像头和显示接口。
# ==========================
# 1. 参数
# ==========================
W, H = 200, 160

ROI_X1, ROI_Y1 = 30, 0
ROI_X2, ROI_Y2 = 170, 160

THRESHOLD = 70
ROAD_PIXEL = 255

RADIUS_STEP = 10       # Δr：15 -> 10
ANGLE_STEP = 1.0       # 圆周角度步长
MIN_NOISE_PIXELS = 5   # 小于5像素的连通区域舍去
MIN_ARC_LENGTH = 3.0
MAX_ANGLE_DIFF = 20.0  # 相邻正常层的极角变化上限（度）

# 路线预测：只在同一帧内，按半径由近到远追踪一条主路径。
START_ANGLE = 270.0       # 图像正前方：x向右、y向下，270度指向上方
START_MAX_ANGLE = 60.0    # 起点只接受正前方左右各60度内的候选
MAX_EMPTY_LAYERS = 1     # 最多跨过1层空层/无可靠匹配层，超过就停止
PREDICT_DISTANCE_RATIO = 1.0  # 位置误差门限 = 半径间隔 × 此系数
MIN_PREDICT_DISTANCE = 8.0    # 位置误差门限下限（像素）
MAX_GAP_ANGLE_DIFF = 30.0     # 跨缺口时极角门限的绝对上限（度）
MAX_DIRECTION_DIFF = 60.0    # 新线段与预测方向的最大夹角（度）
DISTANCE_WEIGHT = 1.0       # 预测位置误差权重
ANGLE_WEIGHT = 0.5          # 与上一个弧心的极角连续性权重
DIRECTION_WEIGHT = 0.7      # 路径行进方向连续性权重

TURN_MODE = "right"         # 路口主路线："right"右转，"left"左转
BRANCH_COLOR = (0, 165, 255) # BGR橙色：显示可连接的其他分支
JUNCTION_WIDTH_RATIO = 1.8  # 弧长相对近期道路宽度明显增大，视为疑似路口
JUNCTION_WIDTH_MARGIN = 8.0 # 额外增长像素，避免近处取整触发
JUNCTION_HOLD_LAYERS = 4    # 疑似路口后允许转向的扫描层数
JUNCTION_MAX_DIRECTION = 125.0  # 允许T字路口从竖直转向横向
ARC_OVERLAP_MARGIN = 8.0    # 分支连接的弧区间角度容差

LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 1


# ==========================
# 2. 删除小噪声连通区域
# ==========================
def remove_small_components(mask):
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_NOISE_PIXELS:
            cleaned[labels == i] = 255

    return cleaned


# ==========================
# 3. 扫描圆周并提取道路弧段
# ==========================
def find_arcs(roi, cx, cy, radius):
    height, width = roi.shape

    step = math.radians(ANGLE_STEP)
    n = int(math.ceil(math.pi / step)) + 1

    angles = np.linspace(math.pi, 2 * math.pi, n)

    xs = np.rint(
        cx + radius * np.cos(angles)
    ).astype(np.int32)

    ys = np.rint(
        cy + radius * np.sin(angles)
    ).astype(np.int32)

    valid = (
        (xs >= 0) & (xs < width) &
        (ys >= 0) & (ys < height)
    )

    road = np.zeros(n, dtype=bool)
    ids = np.flatnonzero(valid)

    road[ids] = (
        roi[ys[ids], xs[ids]] == ROAD_PIXEL
    )

    ids = np.flatnonzero(road)

    if len(ids) == 0:
        return []

    segments = np.split(
        ids,
        np.where(np.diff(ids) > 1)[0] + 1
    )

    arcs = []

    for seg in segments:
        if len(seg) < 2:
            continue

        first = int(seg[0])
        last = int(seg[-1])

        a1 = float(angles[first])
        a2 = float(angles[last])

        if radius * (a2 - a1) < MIN_ARC_LENGTH:
            continue

        am = (a1 + a2) / 2

        mx = int(round(cx + radius * math.cos(am)))
        my = int(round(cy + radius * math.sin(am)))

        # 中点不在道路上时，回退到有效采样点
        if not (
            0 <= mx < width and
            0 <= my < height and
            roi[my, mx] == ROAD_PIXEL
        ):
            mid = int(seg[len(seg) // 2])
            mx, my = int(xs[mid]), int(ys[mid])

        arcs.append((
            mx, my,
            math.degrees(a1),
            math.degrees(a2)
        ))

    return arcs


def arc_angle(arc):
    return (arc[2] + arc[3]) / 2.0


# ==========================
# 4. 轻量路线预测与单主路径选择
# ==========================
def angle_diff(a, b):
    # 兼容角度跨过0/360度的情况。
    return abs((a - b + 180.0) % 360.0 - 180.0)


def predict_next_point(path, radius, cx, cy):
    # path元素为 (实际扫描半径, 实测弧段)。绝不把预测点写入path。
    _, last_arc = path[-1]
    px, py = float(last_arc[0]), float(last_arc[1])

    if len(path) >= 2:
        # 两个最近的已选实测弧心确定局部行进方向。
        old_arc = path[-2][1]
        dx, dy = px - old_arc[0], py - old_arc[1]
    else:
        # 只有一个实测点时，暂用固定圆心指向该点的方向。
        dx, dy = px - cx, py - cy

    length = math.hypot(dx, dy)
    if length < 1e-6:
        dx, dy = 0.0, -1.0
    else:
        dx, dy = dx / length, dy / length

    # 直线外推与下一扫描圆求交，保持预测点位于当前半径层。
    # 半径间隔包含缺失层，所以跨一层时会自动预测到正确距离。
    qx, qy = px - cx, py - cy
    dot = qx * dx + qy * dy
    discriminant = dot * dot + radius * radius - qx * qx - qy * qy
    travel = -dot + math.sqrt(max(0.0, discriminant))
    return (px + travel * dx, py + travel * dy), (dx, dy)


def select_main_path(layers, cx, cy):
    # 每帧重新建立path，不保留上一帧可能过时的路线。
    path = []
    missed = 0

    for index, arcs in enumerate(layers):
        radius = (index + 1) * RADIUS_STEP
        best = None
        best_score = float("inf")

        if not path:
            # 起点：从最近的可靠层中选最接近正前方的弧心。
            # 没有导航指令时，这是一个确定的默认分叉偏好。
            for arc in arcs:
                diff = angle_diff(arc_angle(arc), START_ANGLE)
                if diff <= START_MAX_ANGLE and diff < best_score:
                    best, best_score = arc, diff
        else:
            last_radius, last_arc = path[-1]
            predicted, direction = predict_next_point(path, radius, cx, cy)
            radial_gap = radius - last_radius
            gap_steps = radial_gap / float(RADIUS_STEP)
            distance_limit = max(
                MIN_PREDICT_DISTANCE,
                radial_gap * PREDICT_DISTANCE_RATIO
            )
            angle_limit = min(
                MAX_GAP_ANGLE_DIFF, MAX_ANGLE_DIFF * gap_steps
            )

            for arc in arcs:
                distance = math.hypot(
                    arc[0] - predicted[0], arc[1] - predicted[1]
                )
                polar_diff = angle_diff(arc_angle(arc), arc_angle(last_arc))
                vx, vy = arc[0] - last_arc[0], arc[1] - last_arc[1]
                segment_length = math.hypot(vx, vy)
                if segment_length < 1e-6:
                    continue

                cosine = (vx * direction[0] + vy * direction[1]) / segment_length
                direction_diff = math.degrees(math.acos(
                    max(-1.0, min(1.0, cosine))
                ))

                # 先做硬门限过滤，避免“所有候选都很差”时仍强行接线。
                if (distance > distance_limit or
                        polar_diff > angle_limit or
                        direction_diff > MAX_DIRECTION_DIFF):
                    continue

                # 三项都归一化，防止像素数值压过角度数值。
                score = (
                    DISTANCE_WEIGHT * distance / distance_limit +
                    ANGLE_WEIGHT * polar_diff / angle_limit +
                    DIRECTION_WEIGHT * direction_diff / MAX_DIRECTION_DIFF
                )
                if score < best_score:
                    best, best_score = arc, score

        if best is None:
            # 空层和有候选但全部不可信的层都算丢失。
            # 不添加虚拟点、不更新方向、不在远处另起一条路径。
            missed += 1
            if missed > MAX_EMPTY_LAYERS:
                break
            continue

        path.append((radius, best))
        missed = 0

    return path


# ==========================
# 5. 路口分支：只保留从近处主干可到达的分支
# ==========================
def select_route_tree(layers, cx, cy):
    # 每个节点只接一个最合适的父节点；父节点允许分成左右两支。
    # 预测仍参与评分，路口处单独放宽转向门限，不再强制全图只有一条线。
    nodes = []
    active = []
    start_missed = 0

    for index, arcs in enumerate(layers):
        radius = (index + 1) * RADIUS_STEP
        if not nodes:
            seeds = [a for a in arcs if angle_diff(arc_angle(a), START_ANGLE) <= START_MAX_ANGLE]
            if not seeds:
                start_missed += 1
                if start_missed > MAX_EMPTY_LAYERS:
                    break
                continue
            seed = min(seeds, key=lambda a: angle_diff(arc_angle(a), START_ANGLE))
            width = radius * math.radians(seed[3] - seed[2])
            nodes.append({"path": [(radius, seed)], "parent": None,
                          "widths": [width], "junction_until": -1})
            active = [0]
            continue

        additions = []
        used_parents = set()
        for arc in arcs:
            width = radius * math.radians(arc[3] - arc[2])
            best = None
            best_score = float("inf")
            for parent in active:
                node = nodes[parent]
                last_radius, old = node["path"][-1]
                gap_steps = (radius - last_radius) / float(RADIUS_STEP)
                if gap_steps > MAX_EMPTY_LAYERS + 1:
                    continue
                recent = sorted(node["widths"][-3:])
                baseline = recent[len(recent) // 2]
                widening = (len(node["widths"]) >= 3 and
                            width > baseline * JUNCTION_WIDTH_RATIO + JUNCTION_WIDTH_MARGIN)
                until = max(node["junction_until"],
                            index + JUNCTION_HOLD_LAYERS if widening else -1)
                junction = index <= until
                predicted, direction = predict_next_point(node["path"], radius, cx, cy)
                distance = math.hypot(arc[0] - predicted[0], arc[1] - predicted[1])
                polar_diff = angle_diff(arc_angle(arc), arc_angle(old))
                vx, vy = arc[0] - old[0], arc[1] - old[1]
                length = math.hypot(vx, vy)
                if length < 1e-6:
                    continue
                cosine = (vx * direction[0] + vy * direction[1]) / length
                turn = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                distance_limit = max(MIN_PREDICT_DISTANCE,
                                     (radius - last_radius) * PREDICT_DISTANCE_RATIO)
                angle_limit = min(MAX_GAP_ANGLE_DIFF, MAX_ANGLE_DIFF * gap_steps)
                direction_limit = MAX_DIRECTION_DIFF
                if junction:
                    # 宽弧段可能包含横向道路；后续分裂的弧段可沿左右两侧延伸。
                    # 仍要求弧区间相邻，避免直接跳到远处不相连的黑块。
                    interval_gap = max(0.0, arc[2] - old[3], old[2] - arc[3])
                    if interval_gap > ARC_OVERLAP_MARGIN * gap_steps:
                        continue
                    distance_limit = max(distance_limit, 3.0 * (radius - last_radius) + baseline)
                    angle_limit = 80.0
                    direction_limit = JUNCTION_MAX_DIRECTION
                if (distance > distance_limit or polar_diff > angle_limit or turn > direction_limit):
                    continue
                score = (DISTANCE_WEIGHT * distance / distance_limit +
                         ANGLE_WEIGHT * polar_diff / angle_limit +
                         DIRECTION_WEIGHT * turn / direction_limit)
                if score < best_score:
                    best_score = score
                    best = (parent, until, widening)
            if best is not None:
                parent, until, widening = best
                node = nodes[parent]
                # 宽路口弧长不能反过来污染普通道路的宽度基准。
                widths = node["widths"] if widening or index <= until else (node["widths"] + [width])[-3:]
                additions.append({"path": node["path"] + [(radius, arc)],
                                  "parent": parent, "widths": widths,
                                  "junction_until": until})
                used_parents.add(parent)

        # 只有尚未延伸的分支可以等待一层；已有子节点的父节点及时退出。
        active = [p for p in active if p not in used_parents and
                  radius - nodes[p]["path"][-1][0] <= MAX_EMPTY_LAYERS * RADIUS_STEP]
        for node in additions:
            active.append(len(nodes))
            nodes.append(node)
        if not active:
            break

    if not nodes:
        return [], []
    children = {}
    links = []
    for child, node in enumerate(nodes):
        parent = node["parent"]
        if parent is not None:
            children.setdefault(parent, []).append(child)
            old, arc = nodes[parent]["path"][-1][1], node["path"][-1][1]
            links.append(((old[0], old[1]), (arc[0], arc[1])))

    # 共同主干只画一次；发生分支时，按用户指定方向挑选绿色主路线。
    current = 0
    while current in children:
        options = children[current]
        choose = max if TURN_MODE == "right" else min
        current = choose(options, key=lambda c: arc_angle(nodes[c]["path"][-1][1]))
    return nodes[current]["path"], links


# ==========================
# 6. 主程序
# ==========================
def main():
    cam = camera.Camera(W, H)
    disp = display.Display()

    while not app.need_exit():
        img = cam.read()

        if img is None:
            print("画面读取失败")
            continue

        frame = image.image2cv(
            img, ensure_bgr=True, copy=True
        )

        # --------------------------
        # A. 二值化
        # --------------------------
        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray, (5, 5), 0
        )

        _, binary = cv2.threshold(
            blur, THRESHOLD, 255,
            cv2.THRESH_BINARY_INV
        )

        # 过滤整张二值图中的小连通噪声
        binary = remove_small_components(binary)

        roi = binary[
            ROI_Y1:ROI_Y2,
            ROI_X1:ROI_X2
        ]

        rh, rw = roi.shape

        # --------------------------
        # B. 固定圆心
        # --------------------------
        cx = (rw - 1) / 2.0
        cy = float(rh - 1)

        max_dx = max(cx, rw - 1 - cx)

        max_radius = int(math.ceil(
            math.hypot(max_dx, cy)
        ))

        # --------------------------
        # C. 多半径扫描
        # --------------------------
        layers = []

        for radius in range(
            RADIUS_STEP,
            max_radius + 1,
            RADIUS_STEP
        ):
            arcs = find_arcs(
                roi, cx, cy, radius
            )
            layers.append(arcs)

        # --------------------------
        # D. 同时保留可达左右分支，并选定右转主路线
        # --------------------------
        path, branch_links = select_route_tree(layers, cx, cy)
        links = []
        for i in range(1, len(path)):
            old_arc = path[i - 1][1]
            arc = path[i][1]
            # 若跨过一层缺口，这条线只表示两端实测点之间的插值。
            links.append(((old_arc[0], old_arc[1]), (arc[0], arc[1])))

        # --------------------------
        # E. 最终输出
        # --------------------------
        # 二值图仅用于内部识别；显示彩色原图，并叠加绿色主路径。
        output = frame.copy()

        # 只绘制最终连线，不画辅助圆、候选点、
        # 圆心、ROI边框或调试文字
        output_roi = output[
            ROI_Y1:ROI_Y2,
            ROI_X1:ROI_X2
        ]

        # 先画全部可达分支，再覆盖绿色主路线；两条道路都能看见。
        for p1, p2 in branch_links:
            cv2.line(output_roi, p1, p2, BRANCH_COLOR, LINE_THICKNESS)

        for p1, p2 in links:
            cv2.line(
                output_roi,
                p1, p2,
                LINE_COLOR,
                LINE_THICKNESS
            )

        result = image.cv2image(
            output, bgr=True
        )

        disp.show(result)


if __name__ == "__main__":
    main()
