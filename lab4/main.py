from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ransac_line import fit_line_ransac


TARGET_WIDTH = 960
VALID_LABELS = set(range(19))
RAIL_LABELS = {17, 18}


@dataclass(frozen=True)
class Scene:
    name: str
    image: Path
    mask: Path
    annotation: Path
    roi: tuple[int, int, int, int]
    circle_roi: tuple[int, int, int, int] | None = None
    circle_radius: tuple[int, int] = (8, 180)
    description: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RailSem19 geometry laboratory work")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("scenes.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--scene", action="append", dest="scene_names")
    return parser.parse_args()


def read_manifest(path: Path) -> list[Scene]:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден манифест {path}. Скопируйте scenes.example.json в scenes.json "
            "и укажите четыре фотографии RailSem19 и ROI."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenes: list[Scene] = []
    for item in raw.get("scenes", []):
        required = ("name", "image", "mask", "annotation", "roi")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Сцена {item.get('name', '<без имени>')} не содержит: {', '.join(missing)}")
        roi = tuple(int(value) for value in item["roi"])
        if len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
            raise ValueError(f"Некорректный ROI сцены {item['name']}: {roi}")
        circle_roi = item.get("circle_roi")
        if circle_roi is not None:
            circle_roi = tuple(int(value) for value in circle_roi)
        radius = tuple(int(value) for value in item.get("circle_radius", [8, 180]))
        scenes.append(
            Scene(
                name=str(item["name"]),
                image=Path(item["image"]),
                mask=Path(item["mask"]),
                annotation=Path(item["annotation"]),
                roi=roi,
                circle_roi=circle_roi,
                circle_radius=radius,
                description=str(item.get("description", "")),
            )
        )
    if len(scenes) != 4:
        raise ValueError(f"В манифесте должно быть ровно четыре сцены, найдено: {len(scenes)}")
    return scenes


def resolve_path(path: Path, manifest: Path) -> Path:
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def load_image(path: Path, flags: int) -> np.ndarray:
    image = cv.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image


def resize_to_width(image: np.ndarray, width: int = TARGET_WIDTH) -> tuple[np.ndarray, float]:
    height, old_width = image.shape[:2]
    scale = width / old_width
    new_height = max(1, round(height * scale))
    interpolation = cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR
    return cv.resize(image, (width, new_height), interpolation=interpolation), scale


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv.resize(mask, size, interpolation=cv.INTER_NEAREST)


def crop(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError(f"ROI {roi} выходит за границы изображения {image.shape[1]}x{image.shape[0]}")
    return image[y : y + height, x : x + width]


def normalize_u8(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max(initial=0))
    if maximum == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.uint8(np.clip(values * (255.0 / maximum), 0, 255))


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    result = np.zeros_like(magnitude, dtype=np.float32)
    angle = direction % 180.0
    for y in range(1, magnitude.shape[0] - 1):
        for x in range(1, magnitude.shape[1] - 1):
            current_angle = angle[y, x]
            if current_angle < 22.5 or current_angle >= 157.5:
                first, second = magnitude[y, x - 1], magnitude[y, x + 1]
            elif current_angle < 67.5:
                first, second = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            elif current_angle < 112.5:
                first, second = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                first, second = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            if magnitude[y, x] >= first and magnitude[y, x] >= second:
                result[y, x] = magnitude[y, x]
    return result


def hysteresis(nms: np.ndarray, low_ratio: float = 0.1, high_ratio: float = 0.2) -> np.ndarray:
    maximum = float(nms.max(initial=0))
    if maximum == 0:
        return np.zeros(nms.shape, dtype=np.uint8)
    low, high = low_ratio * maximum, high_ratio * maximum
    strong = nms >= high
    accepted = strong.copy()
    queue = [(int(y), int(x)) for y, x in zip(*np.nonzero(strong))]
    cursor = 0
    while cursor < len(queue):
        y, x = queue[cursor]
        cursor += 1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < nms.shape[0] and 0 <= xx < nms.shape[1] and not accepted[yy, xx]:
                    if nms[yy, xx] >= low:
                        accepted[yy, xx] = True
                        queue.append((yy, xx))
    return np.where(accepted, 255, 0).astype(np.uint8)


def custom_canny(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    smooth = cv.GaussianBlur(gray, (5, 5), 1)
    gx = cv.Sobel(smooth, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(smooth, cv.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    direction = np.degrees(np.arctan2(gy, gx)) % 180.0
    nms = non_maximum_suppression(magnitude, direction)
    return smooth, magnitude, nms, hysteresis(nms)


def contour_data(binary: np.ndarray) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted((contour for contour in contours if cv.contourArea(contour) > 0), key=cv.contourArea, reverse=True)[:3]
    data = []
    for index, contour in enumerate(contours, 1):
        area = float(cv.contourArea(contour))
        perimeter = float(cv.arcLength(contour, True))
        moments = cv.moments(contour)
        centroid = None if moments["m00"] == 0 else [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        x, y, width, height = cv.boundingRect(contour)
        (center, size, angle) = cv.minAreaRect(contour)
        data.append({"index": index, "area": area, "perimeter": perimeter, "centroid": centroid, "bounding_rect": [x, y, width, height], "min_area_rect": [list(center), list(size), angle]})
    return contours, data


def line_length(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line.reshape(4)
    return float(math.hypot(x2 - x1, y2 - y1))


def choose_rail_lines(lines: np.ndarray | None, shape: tuple[int, int]) -> list[np.ndarray]:
    if lines is None:
        return []
    height, width = shape
    candidates = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        length = line_length(line)
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle > 90:
            angle = 180 - angle
        midpoint_y = (y1 + y2) / 2
        if length >= 80 and midpoint_y >= 0.25 * height and 10 <= angle <= 85:
            candidates.append((length + 0.2 * midpoint_y, np.array([[x1, y1, x2, y2]], dtype=np.int32)))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    selected: list[np.ndarray] = []
    for _, line in candidates:
        if len(selected) == 2:
            break
        x1, y1, x2, y2 = line.reshape(4)
        angle = math.atan2(y2 - y1, x2 - x1)
        if all(abs(math.sin(angle - math.atan2(other[0, 3] - other[0, 1], other[0, 2] - other[0, 0]))) > math.sin(math.radians(5)) for other in selected):
            selected.append(line)
    return selected


def draw_ransac_segment(image: np.ndarray, model: np.ndarray, inliers: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if model is None or inliers is None or len(inliers) == 0:
        return None
    a, b, c = map(float, model)
    points = np.asarray(inliers)
    direction = np.array([-b, a], dtype=np.float64)
    projections = points @ direction
    start = direction * projections.min() - c * np.array([a, b])
    end = direction * projections.max() - c * np.array([a, b])
    segment = np.rint(np.vstack((start, end))).astype(np.int32)
    cv.line(image, tuple(segment[0]), tuple(segment[1]), 255, 5)
    return segment, points


def line_mask(shape: tuple[int, int], lines: list[np.ndarray], thickness: int = 5) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        cv.line(mask, tuple(line[0, :2]), tuple(line[0, 2:]), 255, thickness)
    return mask


def metrics(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, int | float]:
    predicted = (prediction > 0) & valid
    expected = (target > 0) & valid
    tp = int(np.count_nonzero(predicted & expected))
    fp = int(np.count_nonzero(predicted & ~expected))
    fn = int(np.count_nonzero(~predicted & expected))
    denominator = tp + fp + fn
    f1_denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / denominator if denominator else 0.0, "f1": 2 * tp / f1_denominator if f1_denominator else 0.0}


def json_rails(annotation: Path, scale: float, roi: tuple[int, int, int, int]) -> list[list[list[int]]]:
    data = json.loads(annotation.read_text(encoding="utf-8"))
    x0, y0, _, _ = roi
    result = []
    for obj in data.get("objects", []):
        if obj.get("label") != "rail":
            continue
        geometry = obj.get("polyline", obj.get("polyline-pair"))
        if geometry is None:
            continue
        if geometry and isinstance(geometry[0][0], (int, float)):
            geometry = [geometry]
        for polyline in geometry:
            result.append([[round(point[0] * scale - x0), round(point[1] * scale - y0)] for point in polyline])
    return result


def save_figure(path: Path, images: list[tuple[np.ndarray, str]], cols: int = 2) -> None:
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    for axis, (image, title) in zip(axes.flat, images):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(images) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_scene(scene: Scene, manifest_path: Path, output: Path) -> list[dict[str, Any]]:
    image_path = resolve_path(scene.image, manifest_path)
    mask_path = resolve_path(scene.mask, manifest_path)
    annotation_path = resolve_path(scene.annotation, manifest_path)
    image = load_image(image_path, cv.IMREAD_COLOR)
    original_mask = load_image(mask_path, cv.IMREAD_UNCHANGED)
    if original_mask.ndim != 2:
        raise ValueError(f"Маска {mask_path} должна быть одноканальной")
    resized, scale = resize_to_width(image)
    resized_mask = resize_mask(original_mask, (resized.shape[1], resized.shape[0]))
    fragment = crop(resized, scene.roi)
    fragment_mask = crop(resized_mask, scene.roi)
    gray = cv.cvtColor(fragment, cv.COLOR_BGR2GRAY)
    smooth, magnitude, nms, own_edges = custom_canny(gray)
    opencv_edges = cv.Canny(smooth, 60, 120, apertureSize=3, L2gradient=True)
    alternate_edges = cv.Canny(smooth, 30, 90, apertureSize=3, L2gradient=True)

    thresholded = cv.adaptiveThreshold(smooth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 5)
    contours, contour_info = contour_data(thresholded)
    contour_overlay = fragment.copy()
    cv.drawContours(contour_overlay, contours, -1, (0, 255, 0), 2)
    for item in contour_info:
        if item["centroid"] is not None:
            center = tuple(np.rint(item["centroid"]).astype(int))
            cv.drawMarker(contour_overlay, center, (0, 0, 255), cv.MARKER_CROSS, 15, 2)
        x, y, width, height = item["bounding_rect"]
        cv.rectangle(contour_overlay, (x, y), (x + width, y + height), (255, 0, 0), 1)
        box = cv.boxPoints(tuple(item["min_area_rect"]))
        cv.polylines(contour_overlay, [np.rint(box).astype(np.int32)], True, (255, 255, 0), 1)
        cv.putText(contour_overlay, str(item["index"]), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    lines = cv.HoughLinesP(own_edges, 1, np.pi / 180, 40, minLineLength=80, maxLineGap=20)
    selected_lines = choose_rail_lines(lines, own_edges.shape)
    hough_overlay = fragment.copy()
    if lines is not None:
        for line in lines[:, 0]:
            cv.line(hough_overlay, tuple(line[:2]), tuple(line[2:]), (100, 100, 100), 1)
    for line in selected_lines:
        cv.line(hough_overlay, tuple(line[0, :2]), tuple(line[0, 2:]), (0, 255, 0), 3)

    ys, xs = np.nonzero(own_edges)
    points = np.column_stack((xs, ys))
    model, inliers = fit_line_ransac(points, tolerance=3, attempts=100)
    ransac_overlay = fragment.copy()
    segment_data = draw_ransac_segment(ransac_overlay, model, inliers)
    if segment_data is not None:
        segment, supporting_points = segment_data
        ransac_overlay[supporting_points[:, 1], supporting_points[:, 0]] = (0, 0, 255)
        cv.line(ransac_overlay, tuple(segment[0]), tuple(segment[1]), (0, 255, 0), 3)

    target = np.isin(fragment_mask, list(RAIL_LABELS))
    valid = np.isin(fragment_mask, list(VALID_LABELS))
    hough_prediction = line_mask(own_edges.shape, selected_lines)
    ransac_prediction = np.zeros_like(own_edges)
    if segment_data is not None:
        cv.line(ransac_prediction, tuple(segment_data[0][0]), tuple(segment_data[0][1]), 255, 5)
    metric_rows = []
    for method, prediction in (("canny", own_edges), ("hough", hough_prediction), ("ransac", ransac_prediction)):
        metric_rows.append({"scene": scene.name, "method": method, **metrics(prediction, target, valid)})

    scene_dir = output / scene.name
    scene_dir.mkdir(parents=True, exist_ok=True)
    save_figure(scene_dir / "canny_pipeline.png", [(fragment, "Фрагмент"), (smooth, "Гаусс 5x5"), (normalize_u8(magnitude), "Величина градиента"), (normalize_u8(nms), "NMS"), (own_edges, "Собственный Канни"), (opencv_edges, "OpenCV Canny 60/120"), (alternate_edges, "OpenCV Canny 30/90")])
    save_figure(scene_dir / "contours.png", [(thresholded, "Адаптивная маска"), (contour_overlay, "Контуры и признаки")])
    save_figure(scene_dir / "lines.png", [(hough_overlay, "Все линии и выбранные кандидаты"), (ransac_overlay, "RANSAC")])
    save_figure(scene_dir / "comparison.png", [(fragment, "Фрагмент"), (target.astype(np.uint8) * 255, "RailSem19: классы 17 и 18"), (own_edges, "Канни"), (hough_prediction, "Хаф"), (ransac_prediction, "RANSAC")])

    if scene.circle_roi is not None:
        circle_fragment = crop(resized, scene.circle_roi)
        circle_gray = cv.cvtColor(circle_fragment, cv.COLOR_BGR2GRAY)
        circle_smooth = cv.medianBlur(circle_gray, 5)
        circles = cv.HoughCircles(circle_smooth, cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=100, param2=30, minRadius=scene.circle_radius[0], maxRadius=scene.circle_radius[1])
        circle_overlay = circle_fragment.copy()
        if circles is not None:
            for cx, cy, radius in np.rint(circles[0]).astype(int):
                cv.circle(circle_overlay, (cx, cy), radius, (0, 255, 0), 2)
                cv.circle(circle_overlay, (cx, cy), 2, (0, 0, 255), 3)
        save_figure(scene_dir / "circles.png", [(circle_fragment, "Деталь"), (circle_overlay, "HoughCircles")])

    polylines = json_rails(annotation_path, scale, scene.roi)
    annotation_overlay = fragment.copy()
    for polyline in polylines:
        if len(polyline) > 1:
            cv.polylines(annotation_overlay, [np.asarray(polyline, dtype=np.int32)], False, (255, 0, 255), 2)
    save_figure(scene_dir / "annotation_overlay.png", [(annotation_overlay, "JSON rail и фрагмент"), (hough_overlay, "Выбранные линии Хафа"), (ransac_overlay, "RANSAC")])
    (scene_dir / "contour_features.json").write_text(json.dumps(contour_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (scene_dir / "scene_info.json").write_text(json.dumps({"description": scene.description, "roi": scene.roi, "scale": scale, "hough_detected": 0 if lines is None else len(lines), "hough_selected": len(selected_lines), "ransac_inliers": 0 if inliers is None else len(inliers), "json_rail_polylines": len(polylines)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return metric_rows


def write_report(output: Path, rows: list[dict[str, Any]], scenes: list[Scene]) -> None:
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene", "method", "tp", "fp", "fn", "iou", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    table = ["| Сцена | Метод | TP | FP | FN | IoU | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    table.extend(f"| {row['scene']} | {row['method']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['iou']:.4f} | {row['f1']:.4f} |" for row in rows)
    scene_list = "\n".join(f"- **{scene.name}** — {scene.description or 'описание не задано'}; ROI `{scene.roi}`." for scene in scenes)
    report = f"""# Лабораторная работа №4

## Выбранные сцены

{scene_list}

## Численное сравнение

{chr(10).join(table)}

Метрики рассчитаны только на допустимых индексах классов. Эталон металла рельсов — объединение классов 17 и 18. Низкие IoU и F1 не означают автоматически плохое положение линии: Канни выделяет границы, а Хаф и RANSAC представлены тонкими линиями шириной 5 пикселей, тогда как эталонная маска покрывает поверхность металла.

## Методика и ограничения

- Фотографии уменьшены до ширины 960 пикселей, маски — через `INTER_NEAREST`; один заранее выбранный ROI используется всеми методами.
- Собственный Канни использует Gaussian 5x5, Sobel 3x3, ручной NMS по четырём секторам и очередь 8-связной гистерезисной проверки.
- Хаф запускается с параметрами 1 пиксель, 1 градус, 40 голосов, длина 80 и разрыв 20. Кандидаты выбираются только по геометрии изображения: длина, положение в нижней части ROI и различие направлений; эталон при выборе не используется.
- RANSAC получает все ненулевые точки собственного Канни с `tolerance=3` и `attempts=100`, поэтому одна прямая описывает лишь один локальный участок кривого или разветвлённого пути.
- Контуры получены через `RETR_EXTERNAL` и `CHAIN_APPROX_SIMPLE`; сохраняются не более трёх крупнейших контуров с ненулевой площадью.

## Ответы на вопросы

1. Знак градиента показывает направление изменения яркости, а вещественный тип не допускает переполнения при вычислении Собеля и `hypot`.
2. NMS сравнивает точки вдоль градиента, чтобы оставить локальный максимум поперёк границы и сделать её тонкой.
3. Слабая граница может быть связана с сильной цепочкой через несколько пикселей, поэтому нужен итерационный обход очередью, а не один проход.
4. Площадь многоугольника — геометрическая величина, а число белых пикселей зависит от дискретизации, толщины и формы границы.
5. `rho` — расстояние от начала координат до прямой, `theta` — угол нормали к ней в параметрическом пространстве Хафа.
6. Да. RANSAC выбирает модель с наибольшей поддержкой среди входных точек; длинная посторонняя граница или шум могут дать большую поддержку.
7. На повороте рельсы меняют направление, а на стрелке появляются несколько ветвей, поэтому две глобальные прямые теряют часть структуры или описывают не тот участок.
8. Класс 12 — пространство между рельсами, классы 17 и 18 — поверхности металлических рельсов (`rail-raised` и `rail-embedded`); для этой работы используются только 17 и 18.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        scenes = read_manifest(args.manifest)
        if args.scene_names:
            names = set(args.scene_names)
            scenes = [scene for scene in scenes if scene.name in names]
            if len(scenes) != len(names):
                raise ValueError("В --scene указано имя, которого нет в манифесте")
        args.output.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for scene in scenes:
            rows.extend(process_scene(scene, args.manifest, args.output))
        write_report(args.output, rows, scenes)
        print(f"Готово: результаты сохранены в {args.output.resolve()}")
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
