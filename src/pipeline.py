"""Filtering, noise reduction & sharpening -> OCR comparison for diploma-number images.

Usage:
    python src/pipeline.py --input data/raw --output results --gt 571012022000056
Input images: image1.jpeg ... imageN.jpeg (scans rotated 90 deg; use --rotate 0 if upright).
"""
import argparse, glob, io, json, os, re
import cv2, numpy as np, pytesseract
from PIL import Image


def lcs_accuracy(ocr_text, gt):
    o, g = re.sub(r"\D", "", ocr_text), re.sub(r"\D", "", gt)
    dp = [[0] * (len(g) + 1) for _ in range(len(o) + 1)]
    for i in range(1, len(o) + 1):
        for j in range(1, len(g) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if o[i-1] == g[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1], len(g), 100 * dp[-1][-1] / len(g)


def crop_number(path, rotate):
    im = Image.open(path).convert("RGB")
    if rotate:
        im = im.rotate(-rotate, expand=True)
        buf = io.BytesIO(); im.save(buf, "JPEG"); buf.seek(0)  # same JPEG re-encode as the report
        im = Image.open(buf).convert("RGB")
    w, h = im.size
    return cv2.cvtColor(np.array(im.crop((0, int(h*.90), int(w*.40), int(h*.99)))), cv2.COLOR_RGB2GRAY)


def enhance(gray):
    return {
        "Original": gray,
        "Mean Filter": cv2.blur(gray, (3, 3)),
        "Median Filter": cv2.medianBlur(gray, 3),
        "Gaussian Filter": cv2.GaussianBlur(gray, (3, 3), 0),
        "Sharpening": cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0),
    }


def ocr(img):
    up = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(bw, config="--psm 7").strip()


def add_salt_pepper(gray, amount=0.08, seed=42):
    np.random.seed(seed)
    out = gray.copy(); h, w = out.shape; n = int(amount * h * w * 0.5)
    ys = np.random.randint(0, h, n); xs = np.random.randint(0, w, n); out[ys, xs] = 255
    ys = np.random.randint(0, h, n); xs = np.random.randint(0, w, n); out[ys, xs] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw"); ap.add_argument("--output", default="results")
    ap.add_argument("--gt", default="571012022000056"); ap.add_argument("--rotate", type=int, default=90)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.input, "*.jp*g")) + glob.glob(os.path.join(a.input, "*.png")),
                   key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1]))
    results = {}
    for idx, f in enumerate(files, 1):
        gray = crop_number(f, a.rotate)
        results[f"image{idx}"] = {}
        for name, im in enhance(gray).items():
            txt = ocr(im); c, t, acc = lcs_accuracy(txt, a.gt)
            results[f"image{idx}"][name] = dict(ocr=txt, correct=c, total=t, accuracy=round(acc, 2))
            cv2.imwrite(os.path.join(a.output, f"img{idx}_{name.replace(' ', '_')}.png"), im)
        print(idx, {k: v["accuracy"] for k, v in results[f"image{idx}"].items()})
    base = max((crop_number(f, a.rotate) for f in files), key=lambda g: g.size)
    noisy = add_salt_pepper(base)
    e = enhance(noisy)
    med = e["Median Filter"]
    sp = {"Noisy": noisy, "Mean Filter": e["Mean Filter"], "Median Filter": med, "Gaussian Filter": e["Gaussian Filter"],
          "Median + Sharpening": cv2.addWeighted(med, 1.5, cv2.GaussianBlur(med, (0, 0), 3), -0.5, 0)}
    sp_res = {}
    for name, im in sp.items():
        txt = ocr(im); c, t, acc = lcs_accuracy(txt, a.gt)
        sp_res[name] = dict(ocr=txt, correct=c, total=t, accuracy=round(acc, 2))
        cv2.imwrite(os.path.join(a.output, f"sp_{name.replace(' ', '_').replace('+', 'plus')}.png"), im)
    print("salt&pepper", {k: v["accuracy"] for k, v in sp_res.items()})
    json.dump({"main": results, "salt_pepper": sp_res}, open(os.path.join(a.output, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
