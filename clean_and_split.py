# -*- coding: utf-8 -*-
"""
Hierarchical Chunking (Leaf-only) + Full Contextual Leaf Enrichment
- Cây: Điều -> Khoản -> Điểm -> Bullet
- Chỉ emit LEAF (và LEAF_WINDOW nếu quá dài) để index/embedding
- Mỗi leaf có:
    * text: nội dung nguyên bản của leaf
    * enriched_text: [CHAPTER] + [ARTICLE] + [CLAUSE] + [POINT] (full context) + leaf text
    * rerank_title/rerank_body: breadcrumb + head + text (cho BM25 & rerank/hiển thị)
- KHÔNG tạo join-map, node cha chỉ xuất khi debug (no text)
"""

import re, json, os, unicodedata
from pathlib import Path

# ===================== CONFIG =====================
INPUTS = [
    ("Nghị định 168/2024/NĐ-CP", "data/raw/168-2024-ND-CP.txt"),
    ("Luật TT, ATGT 2024", "data/raw/luat_trat_tu_an_toan_giaothong_duongbo.txt"),
    ("Luật Đường bộ 2024", "data/raw/luatduongbo.txt"),
]
OUT_DIR = Path("data/processed"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Leaf window (CPU-friendly)
MAX_TOKENS_LEAF = 1500     # <— theo yêu cầu
WIN_TOK = 900
OVERLAP_TOK = 300

# Context rút gọn cho clause head
MAX_HEAD_CHARS = 400

# ===================== REGEX =====================
RE_CHAPTER = re.compile(r'^(Chương\s+[IVXLC]+)\.?\s*(.*)$', re.MULTILINE | re.UNICODE)
RE_SECTION = re.compile(r'^(Mục\s+\d+)\.?\s*(.*)$', re.MULTILINE | re.UNICODE)
RE_ARTICLE = re.compile(r'^Điều\s+(\d+)\.?\s*(.*)$', re.MULTILINE | re.UNICODE)
RE_CLAUSE  = re.compile(r'^(\d+)\.\s+', re.MULTILINE)
RE_POINT   = re.compile(r'^\s*([a-zA-ZđĐ])\)\s+', re.MULTILINE)
RE_BULLET  = re.compile(r'^\s*[-•]\s+', re.MULTILINE)

# ===================== UTILS =====================
def normalize_text(s: str) -> str:
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r'[ \t\f\v]+', ' ', s)
    s = re.sub(r'\s*\n\s*', '\n', s, flags=re.MULTILINE)
    s = re.sub(r'\n{3,}', '\n\n', s)
    # đảm bảo "Điều X. <title>" có dấu chấm
    s = re.sub(r'^(Điều\s+\d+)(\s+)([^\.\n]*)$', r'\1. \3', s, flags=re.MULTILINE)
    return s.strip()

def token_count(text: str) -> int:
    return len(re.findall(r'\S+', text or ""))

def sliding_windows_by_tokens(text: str, win_tokens=WIN_TOK, overlap_tokens=OVERLAP_TOK):
    toks = re.findall(r'\S+|\s+', text)
    words = [i for i, t in enumerate(toks) if not t.isspace()]
    if not words:
        return [text.strip()] if (text or "").strip() else []
    out, i = [], 0
    step = max(1, win_tokens - overlap_tokens)
    while i < len(words):
        j = min(i + win_tokens, len(words))
        start_tok_idx = words[i]
        end_tok_idx   = words[j-1] + 1
        chunk = "".join(toks[start_tok_idx:end_tok_idx]).strip()
        if chunk:
            out.append(chunk)
        if j >= len(words): break
        i += step
    return out

def find_blocks(regex, text):
    ms = list(regex.finditer(text))
    if not ms:
        return [(0, len(text), None)]
    blocks = []
    for i, m in enumerate(ms):
        start = m.start()
        end   = ms[i+1].start() if i+1 < len(ms) else len(text)
        blocks.append((start, end, m))
    return blocks

def law_code_from_filename(source_file: str) -> str:
    stem = Path(source_file).stem
    m = re.search(r'(\d{1,4}-\d{4})', stem)
    if m: return f"ND{m.group(1)}"
    return stem.upper().replace("-", "")

def build_path(chapter, section, article_no, clause_no=None, point_letter=None, bullet_idx=None):
    parts = []
    if chapter: parts.append(chapter)
    if section: parts.append(section)
    if article_no: parts.append(f"Điều {article_no}")
    if clause_no is not None: parts.append(f"Khoản {clause_no}")
    if point_letter: parts.append(f"Điểm {point_letter}")
    if bullet_idx is not None: parts.append(f"Gạch {bullet_idx}")
    return " > ".join(parts)

def header_of(article_no, clause_no=None, point_letter=None, bullet_idx=None):
    parts = []
    if bullet_idx is not None: parts.append(f"Gạch {bullet_idx}")
    if point_letter: parts.append(f"Điểm {point_letter}")
    if clause_no is not None: parts.append(f"Khoản {clause_no}")
    parts.append(f"Điều {article_no}")
    return " ".join(parts)

def citation_of(law, article_no, clause_no=None, point_letter=None, bullet_idx=None):
    parts = []
    if bullet_idx is not None: parts.append(f"gạch {bullet_idx}")
    if point_letter: parts.append(f"điểm {point_letter}")
    if clause_no is not None: parts.append(f"khoản {clause_no}")
    parts.append(f"Điều {article_no} {law}")
    return " ".join(parts)

def truncate(s: str, max_chars: int) -> str:
    s = re.sub(r'\s+', ' ', (s or '')).strip()
    return s if len(s) <= max_chars else (s[:max_chars].rstrip() + "…")

def extract_clause_head(clause_text: str, max_chars: int = MAX_HEAD_CHARS) -> str:
    # phần đầu khoản: trước điểm a)
    body = re.sub(r'^\d+\.\s+', '', clause_text or '', count=1).strip()
    m = RE_POINT.search(body)
    head = body[:m.start()].strip() if m else body
    head = truncate(head, max_chars)
    return head

# ===== Full contextual enrichment (nội dung CHƯƠNG/ĐIỀU/KHOẢN/ĐIỂM) =====
def enrich_text_full(chapter, article_no, article_title, clause_no, clause_head, point_letter, point_text):
    parts = []
    if chapter:
        # ví dụ: "Chương II. NHỮNG QUY ĐỊNH CHUNG"
        parts.append(f"[CHAPTER] {chapter}")
    if article_no and article_title:
        parts.append(f"[ARTICLE] Điều {article_no}. {article_title}")
    elif article_no:
        parts.append(f"[ARTICLE] Điều {article_no}")
    if clause_no is not None:
        parts.append(f"[CLAUSE] Khoản {clause_no}. {clause_head}" if clause_head else f"[CLAUSE] Khoản {clause_no}")
    if point_letter:
        parts.append(f"[POINT] Điểm {point_letter}) {point_text}" if point_text else f"[POINT] Điểm {point_letter})")
    return "\n".join(parts).strip()

def compose_rerank_text_full(chapter, section, header, clause_head, leaf_text):
    title = " / ".join([t for t in [chapter or "", section or "", header or ""] if t])
    body  = ((clause_head or "") + "\n" + (leaf_text or "")).strip() if clause_head else (leaf_text or "")
    return title, body

# ===================== PARSERS =====================
def parse_articles(doc_text):
    # cắt từ chương đầu tiên
    m_start = RE_CHAPTER.search(doc_text)
    if m_start:
        doc_text = doc_text[m_start.start():]

    for ch_s, ch_e, ch_m in find_blocks(RE_CHAPTER, doc_text):
        ch_block = doc_text[ch_s:ch_e]
        chapter  = (ch_m.group(1) + (". " + ch_m.group(2) if ch_m and ch_m.group(2) else "")) if ch_m else ""
        sections = find_blocks(RE_SECTION, ch_block)

        # không có "Mục": xử lý trực tiếp
        if not sections or (len(sections) == 1 and sections[0][2] is None):
            se_block = ch_block
            section = ""
            art_ms = list(RE_ARTICLE.finditer(se_block))
            for i, am in enumerate(art_ms):
                a_s = am.start()
                a_e = art_ms[i+1].start() if i+1 < len(art_ms) else len(se_block)
                block = se_block[a_s:a_e].strip()
                article_no    = am.group(1).strip()
                article_title = (am.group(2) or "").strip()
                head_end = block.find("\n")
                body = block[head_end+1:].strip() if head_end != -1 else ""
                yield {"chapter": chapter, "section": section, "article_no": article_no,
                       "article_title": article_title, "article_text": body}
        else:
            # có mục
            for se_s, se_e, se_m in sections:
                se_block = ch_block[se_s:se_e]
                section  = (se_m.group(1) + (". " + se_m.group(2) if se_m and se_m.group(2) else "")) if se_m else ""
                art_ms = list(RE_ARTICLE.finditer(se_block))
                for i, am in enumerate(art_ms):
                    a_s = am.start()
                    a_e = art_ms[i+1].start() if i+1 < len(art_ms) else len(se_block)
                    block = se_block[a_s:a_e].strip()
                    article_no    = am.group(1).strip()
                    article_title = (am.group(2) or "").strip()
                    head_end = block.find("\n")
                    body = block[head_end+1:].strip() if head_end != -1 else ""
                    yield {"chapter": chapter, "section": section, "article_no": article_no,
                           "article_title": article_title, "article_text": body}

def split_clauses(article_text):
    ms = list(RE_CLAUSE.finditer(article_text or ""))
    if not ms:
        return [(None, (article_text or "").strip())]
    out = []
    if ms[0].start() > 0:
        pre = (article_text[:ms[0].start()] or "").strip()
        if pre:
            out.append((None, pre))
    for i, m in enumerate(ms):
        s = m.start(); e = ms[i+1].start() if i+1 < len(ms) else len(article_text)
        out.append((int(m.group(1)), (article_text[s:e] or "").strip()))
    return out

def split_points(clause_text):
    body = re.sub(r'^\d+\.\s+', '', clause_text or '', count=1).strip()
    ms = list(RE_POINT.finditer(body))
    if not ms:
        return []
    out = []
    for i, m in enumerate(ms):
        s = m.start(); e = ms[i+1].start() if i+1 < len(ms) else len(body)
        letter = m.group(1).lower()
        pt = (body[s:e] or "").strip()
        pt = re.sub(r'^\s*[a-zA-ZđĐ]\)\s+', '', pt)
        out.append((letter, pt))
    return out

def split_bullets(text):
    ms = list(RE_BULLET.finditer(text or ""))
    if not ms:
        return []
    bullets = []
    for i, m in enumerate(ms):
        s = m.start(); e = ms[i+1].start() if i+1 < len(ms) else len(text)
        bt = (text[s:e] or "").strip()
        bt = re.sub(r'^\s*[-•]\s+', '', bt)
        bullets.append(bt)
    return bullets

# ===================== EMIT LEAF =====================
def emit_leaf(items, *, law, source_file, chapter, section,
              article_no, article_title, clause_no, point_letter,
              bullet_idx, clause_head, text):
    law_code = law_code_from_filename(source_file)
    base_id = f"{Path(source_file).stem}_D{article_no}"
    if clause_no is not None:
        base_id += f"_K{clause_no}"
    if point_letter:
        base_id += f"_{point_letter}"
    if bullet_idx is not None:
        base_id += f"_b{bullet_idx}"

    header = header_of(article_no, clause_no, point_letter, bullet_idx)
    display_citation = citation_of(law, article_no, clause_no, point_letter, bullet_idx)
    path = build_path(chapter, section, article_no, clause_no, point_letter, bullet_idx)

    # === Full contextual enrichment cho EMBEDDING ===
    # Nếu là bullet, point_text = None (vì text là nội dung bullet)
    # Nếu là điểm, point_text = None nếu có bullet, hoặc = text nếu là leaf cuối
    point_text_for_enrich = None if bullet_idx is not None else (text if point_letter else None)
    enriched = enrich_text_full(chapter, article_no, article_title, clause_no, clause_head, point_letter, point_text_for_enrich)
    
    # Thêm nội dung leaf vào cuối nếu chưa có
    if text and (bullet_idx is not None or not point_letter):
        enriched = (enriched + "\n" + text).strip()

    # === Văn bản giàu cho RERANK/HIỂN THỊ (KHÔNG embed) ===
    rerank_title, rerank_body = compose_rerank_text_full(chapter, section, header, clause_head, text)

    def add_leaf_record(_id, _enriched):
        items.append({
            "id": _id,
            "granularity": "leaf" if "_w" not in _id else "leaf_window",
            "law": law,
            "law_code": law_code,
            "chapter": chapter,
            "section": section,
            "article_no": article_no,
            "article_title": article_title,
            "clause_no": str(clause_no) if clause_no is not None else None,
            "point": point_letter,
            "bullet_idx": bullet_idx,
            "header": header,
            "display_citation": display_citation,
            "path": path,
            "path_text": path,                 # breadcrumb cho BM25
            "clause_head": clause_head,        # đầu khoản rút gọn
            "text": (text or "").strip(),      # leaf gốc
            "enriched_text": _enriched.strip(),# dùng để EMBED (full context)
            "rerank_title": rerank_title,      # cho BM25 & rerank
            "rerank_body": rerank_body,        # cho BM25 & rerank
            "source_file": os.path.basename(source_file)
        })

    # Nếu enriched quá dài, cắt window theo tokens
    if token_count(enriched) > MAX_TOKENS_LEAF:
        for i, w in enumerate(sliding_windows_by_tokens(enriched, WIN_TOK, OVERLAP_TOK), 1):
            add_leaf_record(f"{base_id}_w{i}", w)
    else:
        add_leaf_record(base_id, enriched)

# ===================== PIPELINE =====================
def process_one(law_name: str, path: str):
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    doc = normalize_text(raw)

    items = []

    for art in parse_articles(doc):
        chapter = art["chapter"]; section = art["section"]
        article_no = art["article_no"]; article_title = art["article_title"]
        article_text = art["article_text"]

        # level 1: Khoản
        clauses = split_clauses(article_text)
        for clause_no, clause_body in clauses:
            clause_head = extract_clause_head(clause_body) if clause_no is not None else ""

            # level 2: Điểm
            points = split_points(clause_body)
            if points:
                for letter, ptext in points:
                    bullets = split_bullets(ptext)
                    if bullets:
                        for bi, bt in enumerate(bullets, 1):
                            emit_leaf(
                                items,
                                law=law_name, source_file=path,
                                chapter=chapter, section=section,
                                article_no=article_no, article_title=article_title,
                                clause_no=clause_no, point_letter=letter, bullet_idx=bi,
                                clause_head=clause_head, text=bt
                            )
                    else:
                        emit_leaf(
                            items,
                            law=law_name, source_file=path,
                            chapter=chapter, section=section,
                            article_no=article_no, article_title=article_title,
                            clause_no=clause_no, point_letter=letter, bullet_idx=None,
                            clause_head=clause_head, text=ptext
                        )
            else:
                # Không có điểm -> leaf là Khoản (trừ preamble None)
                if clause_no is not None:
                    emit_leaf(
                        items,
                        law=law_name, source_file=path,
                        chapter=chapter, section=section,
                        article_no=article_no, article_title=article_title,
                        clause_no=clause_no, point_letter=None, bullet_idx=None,
                        clause_head=clause_head, text=clause_body
                    )
                else:
                    preamble = (clause_body or "").strip()
                    if preamble:
                        emit_leaf(
                            items,
                            law=law_name, source_file=path,
                            chapter=chapter, section=section,
                            article_no=article_no, article_title=article_title,
                            clause_no=None, point_letter=None, bullet_idx=None,
                            clause_head="", text=preamble
                        )

    out_path = OUT_DIR / (Path(path).stem + ".json")
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {law_name}: {len(items)} nodes/chunks → {out_path}")

# ===================== MAIN =====================
if __name__ == "__main__":
    print("🔄 Building contextual leaf chunks (leaf-only, full context)…")
    for name, p in INPUTS:
        try:
            if not Path(p).exists():
                print(f"⚠️  Missing: {p}")
                continue
            process_one(name, p)
        except Exception as e:
            print(f"❌ Lỗi xử lý {name} ({p}): {e}")
    print("✅ Done.")
