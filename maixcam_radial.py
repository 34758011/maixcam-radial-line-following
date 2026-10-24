# SPDX-License-Identifier: MIT
"""固定圆心、前半圆径向循迹增强版。仅视觉识别与显示，不发送电机指令。"""
from maix import camera, display, image, app
import cv2
import numpy as np
import math

# ==========================
# 1. 常用参数（优先只调本组）
# ==========================
W, H = 200, 160
ROI_X1, ROI_Y1 = 30, 0
ROI_X2, ROI_Y2 = 170, 160
THRESHOLD = 70             # 灰度<=此值判为黑色道路
ROAD_PIXEL = 255           # 反向二值化后的道路值，不是黑色阈值
RADIUS_STEP = 10
ANGLE_STEP = 1.0
TURN_MODE = "right"        # "right"固定右侧主线；"left"固定左侧主线
SHOW_BRANCHES = True       # 同时显示可达分支
SHOW_DEBUG = False         # 默认只画路线，保持彩色原图
LINE_COLOR = (0, 255, 0)   # BGR绿色：主路线
BRANCH_COLOR = (0, 165, 255)  # BGR橙色：其他分支
LINE_THICKNESS = 1

# ==========================
# 2. 连接、路口与稳定性参数
# ==========================
MIN_NOISE_PIXELS = 5
MIN_ARC_LENGTH = 3.0
START_ANGLE = 270.0
START_MAX_ANGLE = 65.0
MAX_EMPTY_LAYERS = 1       # 每条分支最多跨1层，不无限补线
MIN_PATH_POINTS = 3        # 少于3个实测点不输出可靠路线
MIN_BRANCH_POINTS = 2     # 分叉后至少2个实测点，抑制只出现一层的杂支
BORDER_EXIT_MARGIN = 12   # 分支已接近ROI侧边时，允许只剩一层可见证据
MAX_ARCS_PER_LAYER = 12    # 限制噪声场景计算量
MAX_NODES = 256
MAX_ANGLE_DIFF = 22.0
MAX_GAP_ANGLE_DIFF = 35.0
MAX_DIRECTION_DIFF = 65.0
PREDICT_DISTANCE_RATIO = 1.2
MIN_PREDICT_DISTANCE = 8.0
JUNCTION_WIDTH_RATIO = 1.6
JUNCTION_WIDTH_MARGIN = 4.0
JUNCTION_HOLD_LAYERS = 4
JUNCTION_MAX_DIRECTION = 125.0
ARC_OVERLAP_MARGIN = 10.0
MIN_LINE_SUPPORT = 0.72    # 普通连线至少72%采样落在道路上
MAX_WHITE_RUN = 3         # 普通连线最多穿过3个连续白色采样
GAP_MIN_LINE_SUPPORT = 0.45
HISTORY_WEIGHT = 0.20      # 上帧只做弱评分，不能绕过当前帧硬门限
HISTORY_RESET_FRAMES = 3   # 连续无可靠路径3帧后清除历史
HISTORY_ROOT_SHIFT = 15.0 # 根部变化过大时立即丢弃旧历史
FLOOD_RATIO = 0.80        # ROI几乎全黑时认为阈值/遮挡异常


def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def unit(dx, dy):
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > 1e-6 else (0.0, -1.0)


def arc_angle(arc):
    return (arc[2] + arc[3]) / 2.0


def arc_length(radius, arc):
    return radius * math.radians(arc[3] - arc[2])


def remove_small_components(mask):
    # 一次查表重建，避免逐个连通域反复遍历整张图。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= MIN_NOISE_PIXELS).astype(np.uint8) * 255
    return keep[labels]


class RadialScanner:
    """固定圆心不变：圆周采样坐标只预计算一次，每帧只读取道路值。"""
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.cx, self.cy = (width - 1) / 2.0, float(height - 1)
        maximum = int(math.ceil(math.hypot(max(self.cx, width-1-self.cx), self.cy)))
        self.radii = list(range(RADIUS_STEP, maximum + 1, RADIUS_STEP))
        count = int(math.ceil(180.0 / ANGLE_STEP)) + 1
        angles = np.linspace(math.pi, 2 * math.pi, count)
        self.samples = []
        for radius in self.radii:
            xs = np.rint(self.cx + radius * np.cos(angles)).astype(np.int32)
            ys = np.rint(self.cy + radius * np.sin(angles)).astype(np.int32)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            self.samples.append((angles, xs, ys, valid, np.flatnonzero(valid)))

    def scan(self, roi):
        if roi.shape != (self.height, self.width):
            raise ValueError("ROI尺寸改变后需要重新建立RadialScanner")
        layers = []
        for radius, sample in zip(self.radii, self.samples):
            angles, xs, ys, valid, ids = sample
            road = np.zeros(len(angles), dtype=bool)
            road[ids] = roi[ys[ids], xs[ids]] == ROAD_PIXEL
            selected = np.flatnonzero(road)
            arcs = []
            if len(selected):
                segments = np.split(selected, np.where(np.diff(selected) > 1)[0] + 1)
                for seg in segments:
                    first, last = int(seg[0]), int(seg[-1])
                    a1, a2 = float(angles[first]), float(angles[last])
                    if radius * (a2 - a1) < MIN_ARC_LENGTH:
                        continue
                    # 小半径上许多角度会落到同一像素，不能当成多个独立证据。
                    pixels = xs[seg] + ys[seg] * self.width
                    if len(np.unique(pixels)) < 3:
                        continue
                    mid = (a1 + a2) / 2.0
                    mx = int(round(self.cx + radius * math.cos(mid)))
                    my = int(round(self.cy + radius * math.sin(mid)))
                    if not (0 <= mx < self.width and 0 <= my < self.height and roi[my, mx] == ROAD_PIXEL):
                        center = int(seg[len(seg) // 2])
                        mx, my = int(xs[center]), int(ys[center])
                    arcs.append((mx, my, math.degrees(a1), math.degrees(a2)))
            # 候选过多时优先保留有足够弧长的候选，避免毛刺拖慢主循环。
            if len(arcs) > MAX_ARCS_PER_LAYER:
                arcs = sorted(arcs, key=lambda a: (-arc_length(radius, a), angle_diff(arc_angle(a), START_ANGLE)))[:MAX_ARCS_PER_LAYER]
            layers.append(arcs)
        return layers


class RouteHistory:
    """只记忆上一条可靠实测路线，用于消除评分接近时的抖动。"""
    def __init__(self):
        self.points = {}
        self.misses = 0
        self.mode = TURN_MODE

    def clear(self):
        self.points = {}
        self.misses = 0
        self.mode = TURN_MODE

    def update(self, result):
        if len(result["path"]) >= MIN_PATH_POINTS:
            self.points = {radius: (arc[0], arc[1]) for radius, arc in result["path"]}
            self.misses = 0
        else:
            self.misses += 1
            if self.misses >= HISTORY_RESET_FRAMES:
                self.clear()


def predict_point(node, radius, cx, cy):
    # 沿近期实测方向外推，与目标半径圆求交；不生成伪实测节点。
    px, py = node["arc"][:2]
    dx, dy = node["direction"]
    qx, qy = px - cx, py - cy
    dot = qx * dx + qy * dy
    travel = -dot + math.sqrt(max(0.0, dot * dot + radius * radius - qx*qx - qy*qy))
    return px + travel * dx, py + travel * dy


def line_evidence(roi, p1, p2):
    # 验证整段连接，而不只检查两端。角度接近的孤立黑块不能直接接线。
    count = max(2, int(math.ceil(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))) + 1)
    xs = np.rint(np.linspace(p1[0], p2[0], count)).astype(np.int32)
    ys = np.rint(np.linspace(p1[1], p2[1], count)).astype(np.int32)
    if np.any(xs < 0) or np.any(xs >= roi.shape[1]) or np.any(ys < 0) or np.any(ys >= roi.shape[0]):
        return 0.0, count
    road = roi[ys, xs] == ROAD_PIXEL
    longest, run = 0, 0
    for value in road:
        run = 0 if value else run + 1
        longest = max(longest, run)
    return float(np.count_nonzero(road)) / count, longest


def empty_result(reason="lost"):
    return {"path": [], "links": [], "main_links": [], "status": reason,
            "confidence": 0.0, "error": None, "branches": 0, "nodes": []}


def build_routes(roi, scanner, history=None):
    if TURN_MODE not in ("left", "right"):
        raise ValueError('TURN_MODE只能为"left"或"right"')
    if history is not None and history.mode != TURN_MODE:
        history.clear()
    if float(np.count_nonzero(roi == ROAD_PIXEL)) / roi.size > FLOOD_RATIO:
        return empty_result("mask_flood")
    layers = scanner.scan(roi)
    cx, cy = scanner.cx, scanner.cy
    nodes, active = [], []
    missed_start = 0
    use_history = bool(history is not None and history.points)

    for index, (radius, arcs) in enumerate(zip(scanner.radii, layers)):
        if not nodes:
            seeds = [a for a in arcs if angle_diff(arc_angle(a), START_ANGLE) <= START_MAX_ANGLE]
            if not seeds:
                missed_start += 1
                if missed_start > MAX_EMPTY_LAYERS:
                    break
                continue
            old = history.points.get(radius) if use_history else None
            def seed_score(a):
                temporal = min(1.0, math.hypot(a[0]-old[0], a[1]-old[1]) / HISTORY_ROOT_SHIFT) if old else 0.0
                return angle_diff(arc_angle(a), START_ANGLE) / START_MAX_ANGLE + HISTORY_WEIGHT * temporal
            seed = min(seeds, key=seed_score)
            if old and math.hypot(seed[0]-old[0], seed[1]-old[1]) > HISTORY_ROOT_SHIFT:
                history.clear()
                use_history = False
            nodes.append({"arc": seed, "radius": radius, "parent": None, "depth": 1,
                          # 起点偏左/右不代表道路横着走；启动方向用正前方。
                          "direction": (math.cos(math.radians(START_ANGLE)), math.sin(math.radians(START_ANGLE))),
                          "stable_direction": (math.cos(math.radians(START_ANGLE)), math.sin(math.radians(START_ANGLE))),
                          "widths": [arc_length(radius, seed)], "until": -1,
                          "cost": 0.0, "support": 1.0, "gap": False, "waypoint": None})
            active = [0]
            continue

        additions, used = [], set()
        for arc in arcs:
            width = arc_length(radius, arc)
            best, best_score = None, float("inf")
            for parent in active:
                node = nodes[parent]
                steps = (radius - node["radius"]) / float(RADIUS_STEP)
                if steps > MAX_EMPTY_LAYERS + 1:
                    continue
                old = node["arc"]
                recent = sorted(node["widths"][-3:])
                baseline = recent[len(recent) // 2]
                widening = (node["depth"] >= 3 and len(recent) >= 3 and
                            width > baseline * JUNCTION_WIDTH_RATIO + JUNCTION_WIDTH_MARGIN)
                until = max(node["until"], index + JUNCTION_HOLD_LAYERS if widening else -1)
                junction = index <= until
                px, py = predict_point(node, radius, cx, cy)
                distance = math.hypot(arc[0]-px, arc[1]-py)
                measured = unit(arc[0]-old[0], arc[1]-old[1])
                cosine = measured[0]*node["direction"][0] + measured[1]*node["direction"][1]
                turn = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                polar = angle_diff(arc_angle(arc), arc_angle(old))
                distance_limit = max(MIN_PREDICT_DISTANCE, steps * RADIUS_STEP * PREDICT_DISTANCE_RATIO)
                angle_limit = min(MAX_GAP_ANGLE_DIFF, MAX_ANGLE_DIFF * steps)
                # 圆心附近极角对横向偏移很敏感；同一条偏侧直线也会快速变角。
                # 启动阶段由空间距离、行进方向和道路证据共同约束，适度放宽极角。
                if node["depth"] < 3:
                    angle_limit = max(angle_limit, 45.0)
                direction_limit = MAX_DIRECTION_DIFF
                if junction:
                    interval_gap = max(0.0, arc[2]-old[3], old[2]-arc[3])
                    if interval_gap > ARC_OVERLAP_MARGIN * steps:
                        continue
                    # 十字路口可能先形成很宽的合并弧；中点到真实出口距离较大。
                    # 根据上一宽弧的范围放宽位置门限，仍必须通过整段道路验证。
                    distance_limit = max(distance_limit, 3.0 * steps * RADIUS_STEP + baseline,
                                         0.5*arc_length(node["radius"], old) + 2.0*steps*RADIUS_STEP)
                    angle_limit, direction_limit = 80.0, JUNCTION_MAX_DIRECTION
                if distance > distance_limit or polar > angle_limit or turn > direction_limit:
                    continue
                support, white_run = line_evidence(roi, old[:2], arc[:2])
                gap = steps > 1.01
                min_support = GAP_MIN_LINE_SUPPORT if gap else MIN_LINE_SUPPORT
                max_white = RADIUS_STEP + MAX_WHITE_RUN if gap else MAX_WHITE_RUN
                waypoint = None
                if junction and not gap and (support < min_support or white_run > max_white):
                    # 窄直角弯：两个弧心直接拉直线会切进白色内角。
                    # 沿可靠入弯方向先走到投影拐点，再接弧心；两段均核验本帧道路。
                    ix, iy = node["stable_direction"]
                    advance = (arc[0]-old[0])*ix + (arc[1]-old[1])*iy
                    direct = math.hypot(arc[0]-old[0], arc[1]-old[1])
                    # 两种拐点分别对应先沿入路方向走、或先横移再恢复入路方向。
                    # 后一种用于宽合并弧分裂后，中点偏侧但直行出口仍然存在的情况。
                    corners = [(int(round(old[0]+advance*ix)), int(round(old[1]+advance*iy))),
                               (int(round(arc[0]-advance*ix)), int(round(arc[1]-advance*iy)))]
                    for q in corners:
                        length1 = math.hypot(q[0]-old[0], q[1]-old[1])
                        length2 = math.hypot(arc[0]-q[0], arc[1]-q[1])
                        if advance <= 0 or min(length1,length2) <= 1 or length1+length2 > 1.6*direct:
                            continue
                        s1, w1 = line_evidence(roi, old[:2], q)
                        s2, w2 = line_evidence(roi, q, arc[:2])
                        if min(s1,s2) >= MIN_LINE_SUPPORT and max(w1,w2) <= MAX_WHITE_RUN:
                            waypoint = q
                            support = (s1*length1+s2*length2)/(length1+length2)
                            white_run = max(w1,w2)
                            break
                if support < min_support or white_run > max_white:
                    continue
                temporal = 0.0
                if use_history and radius in history.points:
                    hx, hy = history.points[radius]
                    temporal = min(1.0, math.hypot(arc[0]-hx, arc[1]-hy) / 20.0)
                # 位置+极角+行进方向+整段道路证据，历史只做小幅加权。
                score = (distance/distance_limit + 0.45*polar/angle_limit +
                         0.70*turn/direction_limit + 0.8*(1.0-support) +
                         0.25*(steps-1.0) + HISTORY_WEIGHT*temporal +
                         0.10*node["cost"]/max(1, node["depth"]))
                if score < best_score:
                    best_score = score
                    best = (parent, measured, until, junction, widening, support, gap, waypoint)
            if best is not None:
                parent, measured, until, junction, widening, support, gap, waypoint = best
                node = nodes[parent]
                # 普通路段平滑方向；路口直接采纳实测转向，避免旧方向阻碍拐弯。
                # 合并宽弧的中点漂移不代表真实转向，先保留入路口方向。
                # 分裂成窄分支后才采纳分支实测方向，避免提前偏向某一侧。
                direction = node["stable_direction"] if widening else measured if junction else unit(
                    0.30*node["direction"][0] + 0.70*measured[0],
                    0.30*node["direction"][1] + 0.70*measured[1])
                widths = node["widths"] if junction else (node["widths"] + [width])[-3:]
                # 弧段刚开始增宽时中点已可能偏移，不污染可靠入弯方向。
                stable = direction if not junction and width <= baseline*1.25+2.0 else node["stable_direction"]
                additions.append({"arc": arc, "radius": radius, "parent": parent,
                                  "depth": node["depth"]+1, "direction": direction,
                                  "stable_direction": stable,
                                  "widths": widths, "until": until,
                                  "cost": node["cost"]+best_score, "support": support, "gap": gap,
                                  "waypoint": waypoint})
                used.add(parent)
        active = [p for p in active if p not in used and
                  radius - nodes[p]["radius"] <= MAX_EMPTY_LAYERS*RADIUS_STEP]
        # 显式限制树大小，防止复杂噪声造成候选无限增长。
        for node in additions:
            if len(nodes) >= MAX_NODES:
                break
            active.append(len(nodes))
            nodes.append(node)
        if not active:
            break

    if not nodes:
        return empty_result()
    children = {}
    for child, node in enumerate(nodes):
        if node["parent"] is not None:
            children.setdefault(node["parent"], []).append(child)
    # 从远到近计算每条支路的持续证据。单层杂支不参与转向选择。
    depth_to_tip = [1] * len(nodes)
    for parent in range(len(nodes)-1, -1, -1):
        if parent in children:
            depth_to_tip[parent] = 1 + max(depth_to_tip[c] for c in children[parent])
    for parent, options in list(children.items()):
        if len(options) > 1:
            def supported_exit(child):
                node = nodes[child]
                x, y = node["arc"][:2]
                side_exit = min(x, scanner.width-1-x) <= BORDER_EXIT_MARGIN
                return side_exit and node["support"] >= 0.9 and not node["gap"]
            reliable = [c for c in options if depth_to_tip[c] >= MIN_BRANCH_POINTS or supported_exit(c)]
            if reliable:
                children[parent] = reliable
            else:
                children[parent] = [min(options, key=lambda c: nodes[c]["cost"])]

    reachable, stack = [], [0]
    while stack:
        parent = stack.pop()
        reachable.append(parent)
        stack.extend(children.get(parent, []))
    links = []
    def edge_links(parent, child):
        first, last = nodes[parent]["arc"][:2], nodes[child]["arc"][:2]
        q, gap = nodes[child]["waypoint"], nodes[child]["gap"]
        return [(first, q, gap), (q, last, gap)] if q is not None else [(first,last,gap)]
    for child in reachable:
        parent = nodes[child]["parent"]
        if parent is not None:
            links.extend(edge_links(parent, child))

    # 方向选择看分支后面数层，不能只比较分裂当层的一个抖动点。
    def representative(child):
        target = child
        for _ in range(2):
            options = children.get(target, [])
            if not options:
                break
            target = max(options, key=lambda c: depth_to_tip[c])
        return nodes[target]

    route, current, forks = [0], 0, 0
    while children.get(current):
        options = children[current]
        if len(options) == 1:
            chosen = options[0]
        else:
            forks += 1
            origin = nodes[current]["arc"]
            dx, dy = nodes[current]["direction"]
            def turn_key(child):
                tip = representative(child)
                vx, vy = unit(tip["arc"][0]-origin[0], tip["arc"][1]-origin[1])
                # 图像y向下，因此正的叉积表示相对当前行进方向向右转。
                signed = math.atan2(dx*vy-dy*vx, dx*vx+dy*vy)
                return signed
            chosen = (max if TURN_MODE == "right" else min)(options, key=turn_key)
        route.append(chosen)
        current = chosen
    if len(route) < MIN_PATH_POINTS:
        return empty_result("insufficient")
    path = [(nodes[c]["radius"], nodes[c]["arc"]) for c in route]
    main_links = []
    for a,b in zip(route,route[1:]):
        main_links.extend(edge_links(a,b))
    quality = sum(nodes[c]["support"] for c in route) / len(route)
    gap_count = sum(int(nodes[c]["gap"]) for c in route)
    confidence = min(1.0, len(route)/8.0) * quality * max(0.4, 1.0-0.12*gap_count)
    # 输出一个近中场偏差，供今后控制层使用；不平滑绘制坐标，避免移出道路。
    near = [(r, a) for r, a in path if r <= 80] or path[:1]
    total = sum(1.0/r for r, _ in near)
    error = (sum(a[0]/r for r, a in near)/total - cx) / max(1.0, cx)
    endpoints = [c for c in reachable if not children.get(c)]
    status = "branch" if any(len(children.get(c, [])) > 1 for c in reachable) else "tracking"
    return {"path": path, "links": links, "main_links": main_links, "status": status,
            "confidence": confidence, "error": error, "branches": len(endpoints), "nodes": nodes}


def analyze(roi, scanner, history=None):
    result = build_routes(roi, scanner, history)
    if history is not None:
        history.update(result)
    return result


def draw_link(output, p1, p2, color, gap):
    if not gap:
        cv2.line(output, p1, p2, color, LINE_THICKNESS)
        return
    # 跨缺失层用虚线，和实际连续检测区别开。
    length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    for start in range(0, int(math.ceil(length)), 6):
        t1, t2 = start/max(1.0, length), min(1.0, (start+3)/max(1.0, length))
        a = (round(p1[0]+(p2[0]-p1[0])*t1), round(p1[1]+(p2[1]-p1[1])*t1))
        b = (round(p1[0]+(p2[0]-p1[0])*t2), round(p1[1]+(p2[1]-p1[1])*t2))
        cv2.line(output, a, b, color, LINE_THICKNESS)


def draw_result(frame, result):
    output = frame.copy()
    out_roi = output[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    if SHOW_BRANCHES:
        for p1, p2, gap in result["links"]:
            draw_link(out_roi, p1, p2, BRANCH_COLOR, gap)
    for p1, p2, gap in result["main_links"]:
        draw_link(out_roi, p1, p2, LINE_COLOR, gap)
    if SHOW_DEBUG:
        cv2.putText(output, "%s q=%.2f" % (result["status"], result["confidence"]),
                    (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
    return output


def main():
    cam = camera.Camera(W, H)
    disp = display.Display()
    scanner = RadialScanner(ROI_X2-ROI_X1, ROI_Y2-ROI_Y1)
    history = RouteHistory()
    while not app.need_exit():
        img = cam.read()
        if img is None:
            # 读帧失败也累计丢失，但不重复显示上一帧的旧路线。
            history.update(empty_result("read_failed"))
            print("画面读取失败")
            continue
        frame = image.image2cv(img, ensure_bgr=True, copy=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        binary = remove_small_components(binary)
        roi = binary[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        result = analyze(roi, scanner, history)
        output = draw_result(frame, result)
        disp.show(image.cv2image(output, bgr=True))


if __name__ == "__main__":
    main()
