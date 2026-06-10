"""
Extract all images from 7 PDF papers into desktop/pictures/.
Usage: python scripts/extract_pdf_images.py
"""
from pathlib import Path
import fitz  # PyMuPDF

PDF_DIR = Path(r"E:\各年级学习资料\大学\本科\大二\科研项目\能带结构分析")
OUTPUT = Path(r"C:\Users\PeiYang\Desktop\pictures")
PDF_NAMES = [
    "1-PRXEn-2025-Tc.pdf",
    "2-ACSEL-2022-Cells.pdf",
    "3-JMCA-2022-SSE.pdf",
    "4-ACEAMI-2025-Semiconductor.pdf",
    "5-EES-2020-Solar.pdf",
    "6-JMCA-2025-ZT.pdf",
    "典型能带结构分析.pdf",
]

OUTPUT.mkdir(parents=True, exist_ok=True)

total = 0
for pdf_name in PDF_NAMES:
    pdf_path = PDF_DIR / pdf_name
    sub = OUTPUT / Path(pdf_name).stem
    sub.mkdir(exist_ok=True)
    doc = fitz.open(str(pdf_path))
    cnt = 0
    for pi in range(len(doc)):
        for img in doc[pi].get_images(full=True):
            info = doc.extract_image(img[0])
            fname = f"p{pi+1:02d}_i{cnt+1:02d}.{info['ext']}"
            fpath = sub / fname
            fpath.write_bytes(info["image"])
            cnt += 1
    doc.close()
    total += cnt
    print(f"{cnt:4d} images  ←  {pdf_name}")

print(f"\nTotal: {total} images extracted to {OUTPUT}")
