from pathlib import Path

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    u, v = np.meshgrid(coords, coords)
    kernel = np.exp(-(u * u + v * v) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def manual_gaussian(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    source = image.astype(np.float64)
    result = source.copy()
    radius = kernel.shape[0] // 2
    height, width = source.shape
    for row in range(radius, height - radius):
        for col in range(radius, width - radius):
            window = source[row - radius : row + radius + 1, col - radius : col + radius + 1]
            result[row, col] = np.sum(window * kernel)
    return result


def save_image(name: str, image: np.ndarray) -> None:
    if not cv.imwrite(str(OUT / name), image):
        raise OSError(f"Не удалось сохранить {name}")


def panel(filename: str, images: list[tuple[str, np.ndarray]], *, cmap: str = "gray",
          vmin: int = 0, vmax: int = 255, cols: int = 3) -> None:
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.3 * rows), squeeze=False)
    for ax, (title, image) in zip(axes.flat, images):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    gray = cv.imread(str(HERE / "image.png"), cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Не удалось открыть image.png")
    height, width = gray.shape
    if (width, height) != (558, 563):
        raise ValueError(f"Ожидался снимок 558x563, получен {width}x{height}")

    # Gaussian kernels and custom filtering versus OpenCV.
    kernels = {(size, sigma): gaussian_kernel(size, sigma)
               for size in (3, 5, 7) for sigma in (1, 2)}
    for size in (3, 5, 7):
        kernel = kernels[(size, 1)]
        print(f"Ядро {size}x{size}, sigma=1:\n{np.array2string(kernel, precision=8)}")
        print(f"Сумма: {kernel.sum():.16f}; симметрия: {np.allclose(kernel, kernel[::-1, ::-1])}")

    custom_images = []
    comparison_rows = []
    for size, sigma in ((3, 1), (3, 2), (7, 1), (7, 2)):
        kernel = kernels[(size, sigma)]
        custom = manual_gaussian(gray, kernel)
        cv_result = cv.GaussianBlur(gray.astype(np.float64), (size, size), sigma, sigmaY=sigma)
        radius = size // 2
        difference = np.abs(custom[radius:height-radius, radius:width-radius]
                            - cv_result[radius:height-radius, radius:width-radius])
        max_diff = float(difference.max())
        comparison_rows.append((size, sigma, max_diff))
        custom_images.append((f"Собственный Gaussian {size}x{size}, σ={sigma}", custom))
        save_image(f"gaussian_manual_{size}_{sigma}.png",
                   np.clip(np.rint(custom), 0, 255).astype(np.uint8))
    print("\nМаксимальная разница с OpenCV (только внутренние пиксели):")
    for size, sigma, diff in comparison_rows:
        print(f"{size}x{size}, σ={sigma}: {diff:.12g}")
    panel("01_manual_gaussian.png", custom_images, cols=2)

    gaussian_noisy = np.clip(np.rint(gray.astype(np.float64) + RNG.normal(0, 20, gray.shape)),
                             0, 255).astype(np.uint8)
    impulse_noisy = gray.copy()
    u = RNG.random(gray.shape)
    impulse_noisy[u < 0.02] = 0
    impulse_noisy[u >= 0.98] = 255
    save_image("gaussian_noisy.png", gaussian_noisy)
    save_image("impulse_noisy.png", impulse_noisy)
    panel("02_noisy_images.png", [("Исходник", gray), ("Гауссовский шум σ=20", gaussian_noisy),
                                   ("Соль и перец: p=0.02 на импульс", impulse_noisy)], cols=3)

    results: dict[str, np.ndarray] = {
        "Гауссовский шум": gaussian_noisy,
        "Импульсный шум": impulse_noisy,
    }
    filter_panels = []
    for noise_name, noisy in (("Гауссовский шум", gaussian_noisy), ("Соль и перец", impulse_noisy)):
        mean_result = cv.blur(noisy, (5, 5))
        gaussian_result = cv.GaussianBlur(noisy, (5, 5), 1)
        median_result = cv.medianBlur(noisy, 5)
        for filter_name, image in (("Среднее", mean_result), ("Гаусс", gaussian_result),
                                   ("Медиана", median_result)):
            label = f"{noise_name} — {filter_name}"
            results[label] = image
            filter_panels.append((label, image))
    panel("03_filters_gaussian_noise.png",
          [("Исходник", gray), ("Шум", gaussian_noisy)] + filter_panels[:3], cols=3)
    panel("04_filters_impulse_noise.png",
          [("Исходник", gray), ("Шум", impulse_noisy)] + filter_panels[3:], cols=3)
    print("\nMSE относительно исходника:")
    for label, image in results.items():
        mse = float(np.mean((gray.astype(np.float64) - image.astype(np.float64)) ** 2))
        print(f"{label}: {mse:.6f}")

    filtered = cv.medianBlur(impulse_noisy, 5)
    _, global_mask = cv.threshold(filtered, 127, 255, cv.THRESH_BINARY_INV)
    otsu_t, otsu_mask = cv.threshold(filtered, 0, 255,
                                     cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    adaptive_mask = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv.THRESH_BINARY_INV, 31, 5)
    print(f"\nПорог Оцу: {otsu_t:g}")
    panel("05_threshold_masks.png", [("После medianBlur 5x5", filtered),
                                      ("Общий порог 127", global_mask),
                                      (f"Оцу T={otsu_t:g}", otsu_mask),
                                      ("Адаптивный Gaussian 31x31, C=5", adaptive_mask)], cols=2)

    kernel = np.ones((3, 3), np.uint8)
    morphology = {
        "Эрозия": cv.erode(adaptive_mask, kernel),
        "Дилатация": cv.dilate(adaptive_mask, kernel),
        "Открытие": cv.morphologyEx(adaptive_mask, cv.MORPH_OPEN, kernel),
        "Закрытие": cv.morphologyEx(adaptive_mask, cv.MORPH_CLOSE, kernel),
    }
    panel("06_morphology.png", [("Адаптивная маска", adaptive_mask), *morphology.items()], cols=3)

    # Show the same detail crop before and after morphology at enlarged scale.
    # The central region includes representative handwriting without changing the source arrays.
    y0, y1 = int(height * 0.35), int(height * 0.65)
    x0, x1 = int(width * 0.25), int(width * 0.75)
    crop_images = [("До обработки", adaptive_mask[y0:y1, x0:x1])]
    crop_images.extend((name, image[y0:y1, x0:x1]) for name, image in morphology.items())
    panel("07_morphology_detail.png", crop_images, cols=3)

    # Opening removes isolated foreground specks, but can break thin strokes;
    # closing may merge nearby characters. Keep the less destructive adaptive mask as final.
    final_mask = adaptive_mask
    save_image("final_mask.png", final_mask)
    print("\nОкончательная маска сохранена без морфологической коррекции:")
    print("открытие может стереть тонкие штрихи, закрытие — слить соседние; "
          "поэтому выбрана исходная адаптивная маска.")
    print(f"Все изображения и графики: {OUT}")


if __name__ == "__main__":
    main()
