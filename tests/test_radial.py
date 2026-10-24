"""Offline geometric tests; no Maix SDK or camera execution."""
import ast
import math
import unittest
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_visual_functions(filename):
    source = filename.read_text(encoding="utf-8")
    compile(source, str(filename), "exec")
    body = [node for node in ast.parse(source).body
            if isinstance(node, (ast.Assign, ast.FunctionDef, ast.ClassDef))
            and not (isinstance(node, ast.FunctionDef) and node.name == "main")]
    scope = {"np": np, "cv2": cv2, "math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(filename), "exec"), scope)
    return scope


def poly(points, width=18):
    result = np.zeros((160, 140), np.uint8)
    cv2.polylines(result, [np.asarray(points, np.int32)], False, 255, width)
    return result


class RadialTests(unittest.TestCase):
    def setUp(self):
        self.api = load_visual_functions(ROOT / "maixcam_radial.py")
        self.scanner = self.api["RadialScanner"](140, 160)

    def run_mask(self, mask, mode="right", history=None):
        self.api["TURN_MODE"] = mode
        return self.api["analyze"](mask, self.scanner, history)

    def t_mask(self):
        mask = poly([(70, 159), (70, 70)])
        cv2.line(mask, (0, 70), (139, 70), 255, 20)
        return mask

    def test_straight(self):
        result = self.run_mask(poly([(70, 159), (70, 0)]))
        self.assertGreaterEqual(len(result["path"]), 14)
        self.assertEqual(result["branches"], 1)
        self.assertLess(abs(result["error"]), 0.1)

    def test_s_curve(self):
        mask = poly([(70,159), (70,130), (90,100), (85,75), (55,45), (65,10)])
        result = self.run_mask(mask)
        self.assertGreaterEqual(len(result["path"]), 13)
        for _, arc in result["path"]:
            self.assertEqual(mask[arc[1], arc[0]], 255)
        for a, b, gap in result["links"]:
            if not gap:
                self.assertGreaterEqual(self.api["line_evidence"](mask, a, b)[0], self.api["MIN_LINE_SUPPORT"])

    def test_t_both_branches_and_mode(self):
        mask = self.t_mask()
        right = self.run_mask(mask)
        left = self.run_mask(mask, "left")
        self.assertEqual(right["status"], "branch")
        self.assertGreaterEqual(right["branches"], 2)
        self.assertGreater(right["path"][-1][1][0], 115)
        self.assertLess(left["path"][-1][1][0], 25)

    def test_y_both_branches(self):
        mask = poly([(70,159), (70,105), (20,25)], 16)
        cv2.line(mask, (70,105), (125,25), 255, 16)
        self.assertGreaterEqual(self.run_mask(mask)["branches"], 2)
        self.assertGreater(self.run_mask(mask)["path"][-1][1][0], 100)
        self.assertLess(self.run_mask(mask, "left")["path"][-1][1][0], 40)

    def test_cross(self):
        mask = self.t_mask()
        cv2.line(mask, (70,70), (70,0), 255, 18)
        result = self.run_mask(mask)
        self.assertGreaterEqual(result["branches"], 3)
        self.assertGreater(result["path"][-1][1][0], 115)

    def test_blank_and_flood(self):
        self.assertFalse(self.run_mask(np.zeros((160,140),np.uint8))["path"])
        self.assertEqual(self.run_mask(np.full((160,140),255,np.uint8))["status"], "mask_flood")

    def test_disconnected_road(self):
        mask = poly([(70,159), (70,0)])
        mask[:100,110:125] = 255
        result = self.run_mask(mask)
        for a, b, _ in result["links"]:
            self.assertLess(max(a[0], b[0]), 95)

    def test_finite_missing_layers(self):
        mask = poly([(70,159), (70,0)])
        yy, xx = np.indices(mask.shape)
        radius = np.hypot(xx-self.scanner.cx, yy-self.scanner.cy)
        mask[np.abs(radius-60) <= 2.5] = 0
        one = self.run_mask(mask)
        self.assertGreaterEqual(one["path"][-1][0], 140)
        self.assertTrue(any(gap for _,_,gap in one["main_links"]))
        mask[np.abs(radius-70) <= 2.5] = 0
        two = self.run_mask(mask)
        self.assertLess(two["path"][-1][0], 60)

    def test_history_not_stale_and_mode_switch(self):
        history = self.api["RouteHistory"]()
        self.run_mask(self.t_mask(), history=history)
        self.assertTrue(history.points)
        left = self.run_mask(self.t_mask(), "left", history)
        self.assertLess(left["path"][-1][1][0], 25)
        for _ in range(3):
            result = self.run_mask(np.zeros((160,140),np.uint8), "left", history)
            self.assertFalse(result["path"])
            self.assertIsNone(result["error"])
        self.assertFalse(history.points)

    def test_color_output_and_no_input_mutation(self):
        frame = np.full((160,200,3), (35,100,180), np.uint8)
        original = frame.copy()
        output = self.api["draw_result"](frame, self.run_mask(self.t_mask()))
        self.assertTrue(np.array_equal(frame, original))
        self.assertTrue(np.array_equal(output[0,0], original[0,0]))
        self.assertTrue(np.any(np.all(output == (0,255,0),axis=2)))
        self.assertTrue(np.any(np.all(output == (0,165,255),axis=2)))

    def test_main_loop_with_software_adapters(self):
        frame = np.full((160,200,3),220,np.uint8)
        frame[:,30:170][self.t_mask() == 255] = 25
        shown = []
        counter = {"exit":0}
        def need_exit():
            counter["exit"] += 1
            return counter["exit"] > 1
        maix = ModuleType("maix")
        maix.camera = SimpleNamespace(Camera=lambda w,h:SimpleNamespace(read=lambda:frame.copy()))
        maix.display = SimpleNamespace(Display=lambda:SimpleNamespace(show=shown.append))
        maix.image = SimpleNamespace(image2cv=lambda img,ensure_bgr,copy:img.copy(),
                                     cv2image=lambda image,bgr:image)
        maix.app = SimpleNamespace(need_exit=need_exit)
        with patch.dict(sys.modules, {"maix":maix}):
            scope = {"__name__":"offline_adapter"}
            source = (ROOT / "maixcam_radial.py").read_text(encoding="utf-8")
            exec(compile(source,"maixcam_radial.py","exec"),scope)
            scope["main"]()
        self.assertEqual(len(shown),1)
        self.assertEqual(shown[0].shape,(160,200,3))
        self.assertTrue(np.any(np.all(shown[0] == (0,255,0),axis=2)))

    def test_geometry_and_preprocessing_sweep(self):
        count = 0
        for width in (10,16,22):
            for offset in (-10,0,10):
                for junction_y in (50,70,90):
                    for kind in ("T","cross","rightL","leftL"):
                        x = 70 + offset
                        mask = poly([(x,159),(x,junction_y)],width)
                        if kind in ("T","cross"):
                            cv2.line(mask,(0,junction_y),(139,junction_y),255,width)
                        else:
                            target = 139 if kind == "rightL" else 0
                            cv2.line(mask,(x,junction_y),(target,junction_y),255,width)
                        if kind == "cross":
                            cv2.line(mask,(x,junction_y),(x,0),255,width)
                        gray = np.where(mask == 255,25,220).astype(np.uint8)
                        _, processed = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0),70,255,cv2.THRESH_BINARY_INV)
                        processed = self.api["remove_small_components"](processed)
                        for variant, image in (("binary",mask),("preprocessed",processed)):
                            with self.subTest(width=width,offset=offset,y=junction_y,kind=kind,variant=variant):
                                result = self.run_mask(image)
                                self.assertTrue(result["path"])
                                tip_x = result["path"][-1][1][0]
                                if kind == "leftL":
                                    self.assertLess(tip_x,25)
                                else:
                                    self.assertGreater(tip_x,115)
                                required = 3 if kind == "cross" else (2 if kind == "T" else 1)
                                self.assertGreaterEqual(result["branches"],required)
                            count += 1
        self.assertEqual(count,216)

    def test_previous_baseline_comparison(self):
        previous = load_visual_functions(ROOT / "baselines/radial_previous.py")
        passed = 0
        for width in (10,16,22):
            for offset in (-10,0,10):
                for y in (50,70,90):
                    for kind in ("T","cross","rightL","leftL"):
                        x = 70+offset
                        mask = poly([(x,159),(x,y)],width)
                        if kind in ("T","cross"):
                            cv2.line(mask,(0,y),(139,y),255,width)
                        else:
                            cv2.line(mask,(x,y),(139 if kind == "rightL" else 0,y),255,width)
                        if kind == "cross":
                            cv2.line(mask,(x,y),(x,0),255,width)
                        layers = [previous["find_arcs"](mask,self.scanner.cx,self.scanner.cy,r) for r in self.scanner.radii]
                        path, links = previous["select_route_tree"](layers,self.scanner.cx,self.scanner.cy)
                        if not path:
                            continue
                        starts = {a for a,b in links}
                        ends = {b for a,b in links}
                        required = 3 if kind == "cross" else (2 if kind == "T" else 1)
                        tip = path[-1][1][0]
                        correct_exit = tip < 25 if kind == "leftL" else tip > 115
                        passed += int(correct_exit and len(ends-starts) >= required)
        # Baseline is frozen so the README's historical development comparison is reproducible.
        self.assertEqual(passed,54)


if __name__ == "__main__":
    unittest.main()
