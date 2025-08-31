# routers/model_api.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Literal, Dict, Any
import io, re, numpy as np
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from pythainlp.tokenize import word_tokenize

router = APIRouter(prefix="/api", tags=["model"])

# ---------- โมเดลโหลดครั้งเดียว ----------
LABSE: SentenceTransformer | None = None
CLIP_PROC: CLIPProcessor | None = None
CLIP_MODEL: CLIPModel | None = None

def startup_models():
    """เรียกจาก main.py ตอนแอปสตาร์ท เพื่อโหลดโมเดลครั้งเดียว"""
    global LABSE, CLIP_PROC, CLIP_MODEL
    if LABSE is None:
        LABSE = SentenceTransformer("sentence-transformers/LaBSE")
    if CLIP_MODEL is None or CLIP_PROC is None:
        CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        CLIP_PROC = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# ---------- Utils: PDF -> text / images ----------
def extract_text_from_pdf(pdf_bytes: bytes, enable_ocr: bool = True) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(p.get_text("text") for p in doc).strip()
    if text or not enable_ocr:
        return text
    # OCR fallback (เรนเดอร์หน้า)
    lines = []
    for p in doc:
        pix = p.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            lines.append(pytesseract.image_to_string(img, lang="tha+eng"))
        except Exception:
            lines.append(pytesseract.image_to_string(img))
    return "\n".join(lines).strip()

def tokenize_thai(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text, engine="newmm")
    return " ".join(tokens)

def extract_images_from_pdf(pdf_bytes: bytes) -> List[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pil_images: List[Image.Image] = []
    for page in doc:
        embedded = page.get_images(full=True)
        if embedded:
            for xref, *_ in embedded:
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
            pil_images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return pil_images

# ---------- Embedding / Similarity ----------
def embed_text_labse(tokenized_text: str) -> np.ndarray:
    assert LABSE is not None
    vec = LABSE.encode(tokenized_text, normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")

def embed_images_clip(images: List[Image.Image]) -> List[np.ndarray]:
    # import ในฟังก์ชันกันกรณีไม่ได้ใช้โหมด image
    import torch  # noqa: F401
    assert CLIP_MODEL is not None and CLIP_PROC is not None
    embs: List[np.ndarray] = []
    for img in images:
        inputs = CLIP_PROC(images=img, return_tensors="pt")
        with torch.no_grad():  # type: ignore
            feats = CLIP_MODEL.get_image_features(**inputs)  # type: ignore
        embs.append(feats.squeeze().cpu().numpy())
    return embs

def compute_lexical_similarity(t1: str, t2: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,3))
    tfidf = vec.fit_transform([t1, t2])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

def compute_hybrid(sem: float, lex: float, alpha=0.5, beta=0.5) -> float:
    return float(alpha * sem + beta * lex)

def average_image_similarity(img_embs_a: List[np.ndarray], img_embs_b: List[np.ndarray]) -> float:
    if not img_embs_a or not img_embs_b: return 0.0
    sims = [float(cosine_similarity([ea], [eb])[0][0]) for ea in img_embs_a for eb in img_embs_b]
    return float(np.mean(sims)) if sims else 0.0

# ---------- (ตัวเลือก) กรอง template ----------
def remove_template_text(student_text: str, template_text: str, threshold=0.9):
    from difflib import SequenceMatcher
    s_lines = student_text.strip().split("\n")
    t_lines = [t.strip() for t in template_text.strip().split("\n") if t.strip()]
    kept = []
    for s in s_lines:
        if any(SequenceMatcher(None, s.strip(), t).ratio() > threshold for t in t_lines):
            continue
        kept.append(s)
    return "\n".join(kept).strip()

# ---------- API ----------
@router.get("/health")
def health():
    return {"ok": True}

@router.post("/score")
async def score_endpoint(
    files: List[UploadFile] = File(...),
    mode: Literal["text","image"] = Query("text"),
    use_template: int = Query(0, ge=0, le=1),
    ocr: int = Query(1, ge=0, le=1),
    alpha: float = Query(0.5),
    beta: float = Query(0.5),
) -> Dict[str, Any]:
    """คำนวณความคล้ายคลึงแบบเป็นคู่ ๆ (text หรือ image)"""
    if len(files) < 2 and not (use_template and len(files) >= 1):
        raise HTTPException(400, "ต้องส่ง PDF อย่างน้อย 2 ไฟล์")

    # อ่านไฟล์ทั้งหมด
    raw_pdfs: Dict[str, bytes] = {f.filename: await f.read() for f in files}

    # template (ถ้ามี)
    template_bytes = None
    for name in list(raw_pdfs.keys()):
        if use_template and name.lower().startswith("template"):
            template_bytes = raw_pdfs.pop(name)
            break

    names = list(raw_pdfs.keys())
    if len(names) < 2:
        raise HTTPException(400, "ต้องส่ง PDF >= 2 ไฟล์ (ไม่นับ template)")

    results: List[Dict[str, Any]] = []

    if mode == "text":
        # เตรียมโมเดล
        startup_models()

        # เตรียมข้อความ
        template_tok = None
        if template_bytes:
            tpl_raw = extract_text_from_pdf(template_bytes, enable_ocr=bool(ocr))
            template_tok = tokenize_thai(tpl_raw)

        clean_texts: Dict[str, str] = {}
        text_vecs: Dict[str, np.ndarray] = {}
        for name in names:
            raw = extract_text_from_pdf(raw_pdfs[name], enable_ocr=bool(ocr))
            if template_tok:
                raw = remove_template_text(raw, template_tok, threshold=0.9)
            tok = tokenize_thai(raw)
            clean_texts[name] = tok
            text_vecs[name] = embed_text_labse(tok)

        # จับคู่คำนวณ
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = names[i], names[j]
                sem = float(cosine_similarity([text_vecs[a]], [text_vecs[b]])[0][0])
                lex = compute_lexical_similarity(clean_texts[a], clean_texts[b])
                hyb = compute_hybrid(sem, lex, alpha, beta)

                if hyb >= 0.85:
                    level, msg = "คล้ายสูง", "ข้อความมีความคล้ายคลึงสูง"
                elif hyb >= 0.65:
                    level, msg = "คล้ายปานกลาง", "ข้อความมีความคล้ายคลึงปานกลาง"
                else:
                    level, msg = "คล้ายน้อย", "ข้อความมีความคล้ายคลึงน้อย"

                results.append({
                    "doc_1": a, "doc_2": b,
                    "hybrid_similarity": hyb,
                    "semantic_similarity": sem,
                    "lexical_similarity": lex,
                    "image_similarity": 0.0,
                    "level": level,
                    "text_message": msg,
                })

    else:  # image mode
        # เตรียมโมเดล
        startup_models()
        img_vecs: Dict[str, List[np.ndarray]] = {}
        for name in names:
            imgs = extract_images_from_pdf(raw_pdfs[name])
            img_vecs[name] = embed_images_clip(imgs) if imgs else []

        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = names[i], names[j]
                score = average_image_similarity(img_vecs[a], img_vecs[b])
                if score >= 0.85:
                    level, msg = "คล้ายสูง", "ภาพ/ไดอะแกรมมีความคล้ายคลึงสูง"
                elif score >= 0.65:
                    level, msg = "คล้ายปานกลาง", "ภาพ/ไดอะแกรมมีความคล้ายคลึงปานกลาง"
                else:
                    level, msg = "คล้ายน้อย", "ภาพ/ไดอะแกรมมีความคล้ายคลึงน้อย"

                results.append({
                    "doc_1": a, "doc_2": b,
                    "hybrid_similarity": score,
                    "semantic_similarity": 0.0,
                    "lexical_similarity": 0.0,
                    "image_similarity": score,
                    "level": level,
                    "text_message": "ไม่มีการประมวลผลข้อความ",
                })

    return {"mode": mode, "files": names, "results": results}