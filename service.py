from __future__ import annotations

import os
import shutil
import uuid
import hashlib
import struct
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from math import erfc, sqrt, log, exp
import base64
import io
from PIL import Image


app = FastAPI(
    title="Photo→Key→Numbers + NIST mini-STS",
    version="2.0.0",
    description=(
        "1) /generate: собрать key из папки с фото и получить random_numbers.txt\n"
        "2) /nist/check-file: прогнать загруженный файл битов через 5 NIST-тестов\n"
        "3) /nist/generate-from-key: сгенерировать числа по key.txt и прогнать NIST-тесты\n"
        "4) /generate-from-canvas: сгенерировать числа из canvas рисунка\n"
        "5) /store-canvas: сохранить canvas как фото и вернуть папку\n"
        "6) /store-photo: сохранить загруженное фото и вернуть папку\n"
        "7) /archive-run: отдать zip с папкой прогона"
    ),
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OUTPUTS_ROOT = Path("outputs")
BATCH_SIZE_DEFAULT = 64

class GenerateRequest(BaseModel):
    photos_folder: str = Field(..., description="Путь к папке с фотографиями")
    low: int = Field(..., description="Нижняя граница диапазона (включительно)")
    high: int = Field(..., description="Верхняя граница диапазона (включительно)")
    count: int = Field(..., description="Сколько чисел сгенерировать")

class GenerateResponse(BaseModel):
    key: str              # путь к key.txt
    random_numbers: str   # путь к random_numbers.txt
    run_dir: str          # путь к главной папке прогона
    numbers: List[int] | None = None  # сгенерированные числа (для удобства фронта)

def list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")
    items = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    items.sort(key=lambda p: p.name.lower())
    if not items:
        raise FileNotFoundError(f"No images found in folder: {folder}")
    return items

def load_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return img

def image_to_cp1251_string(img_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return rgb.reshape(-1, 3).tobytes().decode("cp1251", errors="replace")

def atomic_write_text(path: Path, text: str, encoding: str = "utf-8"):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
    os.replace(tmp, path)

def append_text(path: Path, text: str, encoding: str = "utf-8"):
    with open(path, "a", encoding=encoding, newline="") as f:
        f.write(text)

def new_run_dir() -> Path:
    rid = uuid.uuid4().hex[:12]
    run_dir = OUTPUTS_ROOT / f"run_{rid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def ensure_structure(run_dir: Path) -> tuple[Path, Path, Path]:
    photos_dir = run_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    key_path = run_dir / "key.txt"
    numbers_path = run_dir / "random_numbers.txt"
    return photos_dir, key_path, numbers_path

def copy_all_photos(src_photos: List[Path], dst_dir: Path) -> List[Path]:
    copied = []
    for p in src_photos:
        d = dst_dir / p.name
        shutil.copy2(p, d)
        copied.append(d)
    return copied

def generate_numbers_from_key_fast_sha512(
    key: str, count: int, low: int, high: int, batch_size: int = BATCH_SIZE_DEFAULT
) -> List[int]:
    if low > high:
        raise ValueError("low must be <= high")
    if count <= 0:
        raise ValueError("count must be > 0")

    key_bytes = key.encode("utf-8", errors="ignore")
    base_hasher = hashlib.sha512()
    base_hasher.update(key_bytes)

    numbers: List[int] = []
    counter = 0

    full_range = 1 << 512
    range_size = high - low + 1
    threshold = full_range - (full_range % range_size)

    buf = bytearray(8)
    while len(numbers) < count:
        for _ in range(batch_size):
            buf[0:8] = counter.to_bytes(8, "big")
            counter += 1
            h = base_hasher.copy()
            h.update(buf)
            num = int.from_bytes(h.digest(), "big", signed=False)
            if num < threshold:
                numbers.append(low + (num % range_size))
                if len(numbers) >= count:
                    break
    return numbers

def run_pipeline(photos_folder: Path, low: int, high: int, count: int) -> GenerateResponse:
    run_dir = new_run_dir()
    photos_dir, key_path, numbers_path = ensure_structure(run_dir)

    photos = list_images(photos_folder)
    copied_photos = copy_all_photos(photos, photos_dir)

    atomic_write_text(key_path, "")
    skipped: List[Path] = []

    for p in copied_photos:
        try:
            img_bgr = load_image_bgr(p)
            key_str = image_to_cp1251_string(img_bgr)
            per_key = run_dir / f"key_{p.stem}.txt"
            atomic_write_text(per_key, key_str)
            append_text(key_path, key_str)
            append_text(key_path, "\n")
        except Exception:
            skipped.append(p)

    combined_key = key_path.read_text(encoding="utf-8", errors="ignore")
    numbers = generate_numbers_from_key_fast_sha512(combined_key, count, low, high)
    atomic_write_text(numbers_path, "\n".join(map(str, numbers)))

    return GenerateResponse(
        key=str(key_path.resolve()),
        random_numbers=str(numbers_path.resolve()),
        run_dir=str(run_dir.resolve()),
        numbers=numbers,
    )

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        return run_pipeline(
            photos_folder=Path(req.photos_folder),
            low=req.low,
            high=req.high,
            count=req.count,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def _gammaln(z: float) -> float:
    cof = [76.18009172947146, -86.50532032941677,
           24.01409824083091, -1.231739572450155,
           0.001208650973866179, -5.395239384953e-06]
    x = z
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * log(tmp)
    ser = 1.000000000190015
    for j in range(len(cof)):
        y += 1.0
        ser += cof[j] / y
    return -tmp + log(2.5066282746310005 * ser / x)

def _gser(a: float, x: float) -> float:
    ITMAX = 1000
    eps = 1e-14
    if x <= 0.0:
        return 0.0
    gln = _gammaln(a)
    ap = a
    summ = 1.0 / a
    delt = summ
    for _ in range(ITMAX):
        ap += 1.0
        delt *= x / ap
        summ += delt
        if abs(delt) < abs(summ) * eps:
            return summ * exp(-x + a*log(x) - gln)
    return summ * exp(-x + a*log(x) - gln)

def _gcf(a: float, x: float) -> float:
    ITMAX = 1000
    eps = 1e-14
    FPMIN = 1e-300
    gln = _gammaln(a)
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, ITMAX+1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < eps:
            break
    return h * exp(-x + a*log(x) - gln)

def gammaincc(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        return float("nan")
    if x < a + 1.0:
        P = _gser(a, x)
        return max(0.0, min(1.0, 1.0 - P))
    else:
        Q = _gcf(a, x)
        return max(0.0, min(1.0, Q))

def chi2_sf(x: float, k: int) -> float:
    return gammaincc(k/2.0, x/2.0)

def parse_bits_from_text(text: str) -> List[int]:
    bits = []
    for ch in text:
        if ch == '0': bits.append(0)
        elif ch == '1': bits.append(1)
        elif ch.isspace(): continue
        else:
            raise ValueError(f"Недопустимый символ в файле: {repr(ch)} (нужны только '0' и '1').")
    if not bits:
        raise ValueError("Файл не содержит битов.")
    return bits

def nist_monobit(bits: List[int]) -> Dict[str, Any]:
    n = len(bits)
    S = sum(1 if b == 1 else -1 for b in bits)
    sobs = abs(S) / sqrt(n)
    p = erfc(sobs / sqrt(2.0))
    return {"name":"Frequency (Monobit) Test","n":n,"S":S,"sobs":sobs,"p_value":p,"passed":p >= 0.01}

def nist_block_frequency(bits: List[int], M: int = None) -> Dict[str, Any]:
    n = len(bits)
    if M is None:
        M = max(20, int(0.01*n)+1)
        if n // M >= 100:
            M = max(20, n // 99)
    N = n // M
    if N < 1:
        return {"name":"Frequency Test within a Block","skipped":True,"reason":"слишком короткая последовательность"}
    chi2 = 0.0
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        pi = sum(block)/M
        chi2 += 4.0 * M * (pi - 0.5) ** 2
    p = chi2_sf(chi2, N)
    return {"name":"Frequency Test within a Block","n":n,"M":M,"N_blocks":N,"chi2":chi2,"p_value":p,"passed":p >= 0.01}

def _gf2_rank_32x32(matrix_bits: List[int]) -> int:
    rows = []
    for r in range(32):
        v = 0
        base = r*32
        for c in range(32):
            v = (v << 1) | (matrix_bits[base+c] & 1)
        rows.append(v)
    rank = 0
    mask = 1 << 31
    for _col in range(32):
        pivot = -1
        for r in range(rank, 32):
            if rows[r] & mask:
                pivot = r
                break
        if pivot == -1:
            mask >>= 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(32):
            if r != rank and (rows[r] & mask):
                rows[r] ^= rows[rank]
        rank += 1
        mask >>= 1
        if mask == 0:
            break
    return rank

def nist_binary_matrix_rank(bits: List[int]) -> Dict[str, Any]:
    n = len(bits)
    block = 32*32
    N = n // block
    if N < 1:
        return {"name":"Binary Matrix Rank Test","skipped":True,"reason":"нужно ≥1024 бита"}
    f32 = f31 = 0
    idx = 0
    for _ in range(N):
        r = _gf2_rank_32x32(bits[idx:idx+block])
        idx += block
        if r == 32: f32 += 1
        elif r == 31: f31 += 1
    frest = N - f32 - f31
    p_full, p_m1, p_rest = 0.2888, 0.5776, 0.1336
    chi2 = ((f32 - p_full*N)**2)/(p_full*N) + ((f31 - p_m1*N)**2)/(p_m1*N) + ((frest - p_rest*N)**2)/(p_rest*N)
    p = exp(-chi2/2.0)
    return {"name":"Binary Matrix Rank Test","N_matrices":N,"counts":{"rank_32":f32,"rank_31":f31,"rank_<=30":frest},"chi2":chi2,"p_value":p,"passed":p >= 0.01}

def _longest_run(block: List[int]) -> int:
    best = cur = 0
    for b in block:
        if b:
            cur += 1
            if cur > best: best = cur
        else:
            cur = 0
    return best

def nist_longest_run_ones(bits: List[int]) -> Dict[str, Any]:
    n = len(bits)
    if n < 128:
        return {"name":"Test for the Longest Run of Ones in a Block","skipped":True,"reason":"требуется n ≥ 128"}
    if n < 6272:
        M = 8
        cats = ["<=1","=2","=3",">=4"]
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
        def bin_ix(L):
            if L <= 1: return 0
            if L == 2: return 1
            if L == 3: return 2
            return 3
    elif n < 750000:
        M = 128
        cats = ["<=4","=5","=6","=7","=8",">=9"]
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        def bin_ix(L):
            if L <= 4: return 0
            if L == 5: return 1
            if L == 6: return 2
            if L == 7: return 3
            if L == 8: return 4
            return 5
    else:
        M = 10000
        cats = ["<=10","=11","=12","=13","=14","=15",">=16"]
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        def bin_ix(L):
            if L <= 10: return 0
            if L == 11: return 1
            if L == 12: return 2
            if L == 13: return 3
            if L == 14: return 4
            if L == 15: return 5
            return 6
    N = n // M
    if N < 1:
        return {"name":"Test for the Longest Run of Ones in a Block","skipped":True,"reason":"недостаточно блоков"}
    v = [0]*len(pi)
    for i in range(N):
        L = _longest_run(bits[i*M:(i+1)*M])
        v[bin_ix(L)] += 1
    chi2 = 0.0
    for i in range(len(pi)):
        exp_i = pi[i] * N
        chi2 += (v[i] - exp_i)**2 / exp_i
    dof = len(pi) - 1
    p = chi2_sf(chi2, dof)
    return {"name":"Test for the Longest Run of Ones in a Block","M":M,"N_blocks":N,"counts":v,"categories":cats,"chi2":chi2,"p_value":p,"passed":p >= 0.01}

def _berlekamp_massey(seq: List[int]) -> int:
    n = len(seq)
    c = [0]*n
    b = [0]*n
    c[0] = 1
    b[0] = 1
    L = 0
    m = -1
    for N in range(n):
        d = seq[N]
        for i in range(1, L+1):
            d ^= (c[i] & seq[N - i])
        if d == 1:
            t = c.copy()
            p = N - m
            for j in range(n - p):
                c[j + p] ^= b[j]
            if 2*L <= N:
                L = N + 1 - L
                m = N
                b = t
    return L

def nist_linear_complexity(bits: List[int], M: int = 500) -> Dict[str, Any]:
    n = len(bits)
    if n < M:
        return {"name":"Linear Complexity Test","skipped":True,"reason":f"требуется минимум {M} бит"}
    N = n // M
    if N < 1:
        return {"name":"Linear Complexity Test","skipped":True,"reason":"недостаточно блоков"}
    mu = M/2.0 + (9 + (-1)**(M+1))/36.0 - (M/3.0 + 2/9.0) / (2**M)

    v = [0]*7
    for i in range(N):
        Li = _berlekamp_massey(bits[i*M:(i+1)*M])
        Ti = ((-1)**M) * (Li - mu) + 2.0/9.0
        if   Ti <= -2.5: v[0]+=1
        elif Ti <= -1.5: v[1]+=1
        elif Ti <= -0.5: v[2]+=1
        elif Ti <=  0.5: v[3]+=1
        elif Ti <=  1.5: v[4]+=1
        elif Ti <=  2.5: v[5]+=1
        else:            v[6]+=1
    pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    chi2 = 0.0
    for i in range(7):
        exp_i = pi[i] * N
        chi2 += (v[i] - exp_i)**2 / exp_i
    p = chi2_sf(chi2, 6)
    return {"name":"Linear Complexity Test","M":M,"N_blocks":N,"counts":v,"mu":mu,"chi2":chi2,"p_value":p,"passed":p >= 0.01}

def run_all_tests(bits: List[int]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    results.append(nist_monobit(bits))
    results.append(nist_block_frequency(bits))
    results.append(nist_binary_matrix_rank(bits))
    results.append(nist_longest_run_ones(bits))
    results.append(nist_linear_complexity(bits, M=500))

    passed_count = sum(1 for r in results if not r.get("skipped") and r.get("passed"))
    summary = []
    for r in results:
        item = {"test": r["name"]}
        if r.get("skipped"):
            item.update({"status":"skipped","reason":r.get("reason")})
        else:
            item.update({"status":"passed" if r["passed"] else "failed","p_value":r["p_value"]})
        summary.append(item)
    return {"alpha":0.01,"passed_of_5":passed_count,"summary":summary,"details":results}

BAR_LENGTH = 40
UPDATE_DELAY = 0.05
BATCH_SIZE = 8

def _read_key_from_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Файл {path} не найден. Создайте key.txt.")
    key = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not key:
        raise ValueError(f"Файл {path} пуст — нужен ключ.")
    return key

def generate_numbers_from_key_fast(key: str, count: int, low: int, high: int, batch_size: int = BATCH_SIZE) -> List[int]:
    key_bytes = key.encode("utf-8", errors="ignore")
    base_hasher = hashlib.sha512()
    base_hasher.update(key_bytes)

    numbers: List[int] = []
    counter = 0

    full_range = 1 << 512
    range_size = high - low + 1
    threshold = full_range - (full_range % range_size)

    ctr_buf = bytearray(8)
    pack = struct.pack_into
    start_time = time.time()
    last_update = 0.0

    while len(numbers) < count:
        for _ in range(batch_size):
            pack(">Q", ctr_buf, 0, counter)
            counter += 1
            h = base_hasher.copy()
            h.update(ctr_buf)
            digest = h.digest()
            num = int.from_bytes(digest, "big", signed=False)
            if num < threshold:
                numbers.append(low + (num % range_size))
                if len(numbers) >= count:
                    break
        now = time.time()
        if now - last_update >= UPDATE_DELAY:
            last_update = now
    return numbers

def save_numbers_to_txt(numbers: List[int], filename: Path):
    with open(filename, "w", encoding="utf-8") as f:
        for n in numbers:
            f.write(f"{n}\n")

class GenerateFromKeyReq(BaseModel):
    key_path: str = Field(..., description="Путь до key.txt")
    count: int = Field(1_000_000, ge=1, description="Сколько чисел генерировать")
    low: int = Field(0, description="Нижняя граница диапазона (по умолчанию 0)")
    high: int = Field(1, description="Верхняя граница диапазона (по умолчанию 1)")
    save_to: Optional[str] = Field(None, description="Куда сохранить сгенерированные числа (по одному в строке)")

class GenerateFromCanvasReq(BaseModel):
    canvas_data: str = Field(..., description="Base64 encoded canvas image data")
    low: int = Field(..., description="Нижняя граница диапазона")
    high: int = Field(..., description="Верхняя граница диапазона")
    count: int = Field(..., description="Количество чисел для генерации")

@app.post("/nist/check-file")
async def nist_check_file(file: UploadFile = File(...)):
    try:
        text = (await file.read()).decode("utf-8", errors="strict")
        bits = parse_bits_from_text(text)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    report = run_all_tests(bits)
    return {"total_bits": len(bits), **report}

@app.post("/nist/generate-from-key")
def nist_generate_from_key(req: GenerateFromKeyReq):
    try:
        key = _read_key_from_file(Path(req.key_path))
        if req.low > req.high:
            return JSONResponse(status_code=400, content={"error":"low > high"})
        numbers = generate_numbers_from_key_fast(key, req.count, req.low, req.high)
        if req.low == 0 and req.high == 1:
            bits = numbers
        else:
            bits = [1 if n != 0 else 0 for n in numbers]
        out_path = None
        if req.save_to:
            out_path = str(Path(req.save_to).resolve())
            save_numbers_to_txt(numbers, Path(req.save_to))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    report = run_all_tests(bits)
    return {
        "key_path": str(Path(req.key_path).resolve()),
        "generated_count": len(numbers),
        "saved_to": out_path,
        "total_bits_tested": len(bits),
        **report
    }

@app.post("/generate-from-canvas")
def generate_from_canvas(req: GenerateFromCanvasReq):
    try:
        # Decode base64 image data
        if req.canvas_data.startswith('data:image'):
            # Remove data URL prefix
            base64_data = req.canvas_data.split(',')[1]
        else:
            base64_data = req.canvas_data
        
        image_data = base64.b64decode(base64_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Extract key from image
        key_str = image_to_cp1251_string(img_bgr)
        
        # Generate numbers
        numbers = generate_numbers_from_key_fast_sha512(key_str, req.count, req.low, req.high)
        
        return {
            "numbers": numbers,
            "key_length": len(key_str),
            "range": f"{req.low}-{req.high}",
            "count": len(numbers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing canvas: {str(e)}")

@app.get("/")
def root():
    return {"ok": True, "endpoints": [
        "/generate",
        "/nist/check-file",
        "/nist/generate-from-key",
        "/generate-from-canvas",
        "/store-canvas",
        "/store-photo",
        "/archive-run",
    ]}

# ------------------ Helpers to save inputs and archive runs ------------------

class StoreCanvasReq(BaseModel):
    canvas_data: str
    filename: str | None = None

@app.post("/store-canvas")
def store_canvas(req: StoreCanvasReq):
    try:
        if req.canvas_data.startswith('data:image'):
            base64_data = req.canvas_data.split(',')[1]
        else:
            base64_data = req.canvas_data
        image_data = base64.b64decode(base64_data)

        run_dir = new_run_dir()
        photos_dir, key_path, numbers_path = ensure_structure(run_dir)
        name = (req.filename or f"canvas_{uuid.uuid4().hex[:8]}") + ".jpg"
        out = photos_dir / name

        # Save as JPEG
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image.save(out, format="JPEG", quality=92)

        return {"photos_folder": str(photos_dir.resolve()), "run_dir": str(run_dir.resolve())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to store canvas: {e}")

@app.post("/store-photo")
async def store_photo(file: UploadFile = File(...)):
    try:
        run_dir = new_run_dir()
        photos_dir, key_path, numbers_path = ensure_structure(run_dir)
        suffix = Path(file.filename).suffix or ".jpg"
        name = f"upload_{uuid.uuid4().hex[:8]}{suffix}"
        out = photos_dir / name
        data = await file.read()
        with open(out, "wb") as f:
            f.write(data)
        return {"photos_folder": str(photos_dir.resolve()), "run_dir": str(run_dir.resolve())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to store photo: {e}")

class ArchiveReq(BaseModel):
    run_dir: str

@app.post("/archive-run")
def archive_run(req: ArchiveReq):
    try:
        run_dir = Path(req.run_dir)
        if not run_dir.exists() or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="run_dir not found")
        zip_base = run_dir.with_suffix("")
        zip_path = shutil.make_archive(str(zip_base), 'zip', str(run_dir))
        return FileResponse(zip_path, media_type='application/zip', filename=Path(zip_path).name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to archive: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
