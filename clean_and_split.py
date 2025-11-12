# -*- coding: utf-8 -*-
"""
Lightweight Hierarchical Chunking (CPU-friendly)
- Xây cây cấu trúc: Điều (root) -> Khoản -> Điểm -> Bullet (nếu có)
- Leaf = chunk; thêm metadata path từ gốc đến lá để bảo toàn ngữ cảnh
- Nếu leaf quá dài: cắt sliding-window theo tokens (nhẹ hơn 2048/1024)
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

# Ngưỡng token "nhẹ" cho CPU
MAX_TOKENS_LEAF = 1200       # nếu leaf dài hơn ngưỡng này thì cắt window
WIN_TOK = 800              # kích thước 1 cửa sổ
OVERLAP_TOK = 200           # chồng lấn giữa các cửa sổ

# ===================== REGEX =====================
RE_CHAPTER = re.compile(r'^(Chương\s+[IVXLC]+)\.?\s*(.*)$', re.MULTILINE | re.UNICODE)
RE_SECTION = re.compile(r'^(Mục\s+\d+)\.?\s*(.*)$', re.MULTILINE | re.UNICODE)
RE_ARTICLE = re.compile(r'^Điều\s+(\d+)\.?\s*(.*)$', re.MULTILINE | re.UNICODE)
RE_CLAUSE  = re.compile(r'^(\d+)\.\s+', re.MULTILINE)
RE_POINT   = re.compile(r'^\s*([a-zA-ZđĐ])\)\s+', re.MULTILINE)
# một số văn bản có gạch đầu dòng dạng '-' hoặc '•'
RE_BULLET  = re.compile(r'^\s*[-•]\s+', re.MULTILINE)

# ===================== UTILS =====================
def normalize_text(s: str) -> str:
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r'[ \t\f\v]+', ' ', s)
    s = re.sub(r'\s*\n\s*', '\n', s, flags=re.MULTILINE)
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = re.sub(r'^(Điều\s+\d+)(\s+)([^\.\n]*)$', r'\1. \3', s, flags=re.MULTILINE)
    return s.strip()

def token_count(text: str) -> int:
    # ước lượng token nhẹ: tách theo khoảng trắng
    return len(re.findall(r'\S+', text))

def sliding_windows_by_tokens(text: str, win_tokens=WIN_TOK, overlap_tokens=OVERLAP_TOK):
    toks = re.findall(r'\S+|\s+', text)  # giữ cả khoảng trắng để ghép lại đúng
    # build index of token positions (non-space tokens only)
    words = [i for i, t in enumerate(toks) if not t.isspace()]
    if not words:
        return [text.strip()] if text.strip() else []
    out = []
    i = 0
    step = max(1, win_tokens - overlap_tokens)
    while i < len(words):
        j = min(i + win_tokens, len(words))
        start_tok_idx = words[i]
        end_tok_idx   = words[j-1] + 1  # inclusive last word, so +1
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

# ===================== PARSERS (TREE) =====================
def parse_articles(doc_text):
    """Yield dict(article) with chapter/section/title/text"""
    # cắt từ Chương đầu tiên
    m_start = RE_CHAPTER.search(doc_text)
    if m_start:
        doc_text = doc_text[m_start.start():]

    for ch_s, ch_e, ch_m in find_blocks(RE_CHAPTER, doc_text):
        ch_block = doc_text[ch_s:ch_e]
        chapter  = (ch_m.group(1) + (". " + ch_m.group(2) if ch_m and ch_m.group(2) else "")) if ch_m else ""

        sections = find_blocks(RE_SECTION, ch_block)
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
                yield {
                    "chapter": chapter,
                    "section": section,
                    "article_no": article_no,
                    "article_title": article_title,
                    "article_text": body
                }

def split_clauses(article_text):
    ms = list(RE_CLAUSE.finditer(article_text))
    if not ms:
        return [(None, article_text.strip())]
    out = []
    if ms[0].start() > 0:
        pre = article_text[:ms[0].start()].strip()
        if pre:
            out.append((None, pre))
    for i, m in enumerate(ms):
        s = m.start(); e = ms[i+1].start() if i+1 < len(ms) else len(article_text)
        out.append((int(m.group(1)), article_text[s:e].strip()))
    return out

def split_points(clause_text):
    # bỏ prefix "X. "
    clause_text = re.sub(r'^\d+\.\s+', '', clause_text, count=1).strip()
    ms = list(RE_POINT.finditer(clause_text))
    if not ms:
        return []
    out = []
    for i, m in enumerate(ms):
        s = m.start(); e = ms[i+1].start() if i+1 < len(ms) else len(clause_text)
        letter = m.group(1).lower()
        pt = clause_text[s:e].strip()
        pt = re.sub(r'^\s*[a-zA-ZđĐ]\)\s+', '', pt)  # bỏ "a) "
        out.append((letter, pt))
    return out

def split_bullets(text):
    """Một số điểm/khoản lại có gạch đầu dòng; tách tiếp để thành leaf nhỏ hơn."""
    ms = list(RE_BULLET.finditer(text))
    if not ms:
        return []
    bullets = []
    for i, m in enumerate(ms):
        s = m.start(); e = ms[i+1].start() if i+1 < len(ms) else len(text)
        bt = text[s:e].strip()
        bt = re.sub(r'^\s*[-•]\s+', '', bt)
        bullets.append(bt)
    return bullets

# ===================== EMIT LEAVES =====================
def emit_leaf(items, *, law, source_file, chapter, section,
              article_no, article_title, clause_no, point_letter,
              bullet_idx, text, parents_ids):
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

    # Nếu leaf vượt ngưỡng -> cắt cửa sổ
    if token_count(text) > MAX_TOKENS_LEAF:
        windows = sliding_windows_by_tokens(text, WIN_TOK, OVERLAP_TOK)
        for i, w in enumerate(windows, 1):
            items.append({
                "id": f"{base_id}_w{i}",
                "granularity": "leaf_window",
                "law": law,
                "law_code": law_code,
                "chapter": chapter,
                "section": section,
                "article_no": article_no,
                "article_title": article_title,
                "clause_no": clause_no,
                "point": point_letter,
                "bullet_idx": bullet_idx,
                "header": header,
                "display_citation": display_citation,
                "path": path,
                "parents": parents_ids,
                "text": w,
                "embed_text": w,
                "source_file": os.path.basename(source_file),
                "window_seq": i
            })
    else:
        items.append({
            "id": base_id,
            "granularity": "leaf",
            "law": law,
            "law_code": law_code,
            "chapter": chapter,
            "section": section,
            "article_no": article_no,
            "article_title": article_title,
            "clause_no": clause_no,
            "point": point_letter,
            "bullet_idx": bullet_idx,
            "header": header,
            "display_citation": display_citation,
            "path": path,
            "parents": parents_ids,
            "text": text.strip(),
            "embed_text": text.strip(),
            "source_file": os.path.basename(source_file)
        })

# ===================== PIPELINE =====================
def process_one(law_name: str, path: str):
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    doc = normalize_text(raw)

    items = []
    # Duyệt từng Điều (root node)
    for art in parse_articles(doc):
        chapter = art["chapter"]; section = art["section"]
        article_no = art["article_no"]; article_title = art["article_title"]
        article_text = art["article_text"]

        article_id = f"{Path(path).stem}_D{article_no}"
        # Tạo 1 node root (meta) để join khi cần (không bắt buộc embed)
        items.append({
            "id": article_id,
            "granularity": "article_root",
            "law": law_name,
            "law_code": law_code_from_filename(path),
            "chapter": chapter,
            "section": section,
            "article_no": article_no,
            "article_title": article_title,
            "path": build_path(chapter, section, article_no),
            "parents": {},
            "text": article_text,        # có thể giữ toàn văn Điều để tra cứu chéo
            "embed_text": "",            # không embed toàn văn để nhẹ CPU
            "source_file": os.path.basename(path)
        })

        # level 1: Khoản
        clauses = split_clauses(article_text)
        for clause_no, clause_body in clauses:
            clause_id = f"{article_id}_K{clause_no}" if clause_no is not None else f"{article_id}_PRE"
            items.append({
                "id": clause_id,
                "granularity": "clause_node",
                "law": law_name,
                "law_code": law_code_from_filename(path),
                "chapter": chapter,
                "section": section,
                "article_no": article_no,
                "article_title": article_title,
                "clause_no": clause_no,
                "path": build_path(chapter, section, article_no, clause_no),
                "parents": {"article_id": article_id},
                "text": clause_body,
                "embed_text": "",
                "source_file": os.path.basename(path)
            })

            # level 2: Điểm (nếu có). Nếu không có điểm -> leaf là Khoản.
            points = split_points(clause_body)
            if points:
                for letter, ptext in points:
                    point_id = f"{clause_id}_{letter}"
                    items.append({
                        "id": point_id,
                        "granularity": "point_node",
                        "law": law_name,
                        "law_code": law_code_from_filename(path),
                        "chapter": chapter,
                        "section": section,
                        "article_no": article_no,
                        "article_title": article_title,
                        "clause_no": clause_no,
                        "point": letter,
                        "path": build_path(chapter, section, article_no, clause_no, letter),
                        "parents": {"clause_id": clause_id, "article_id": article_id},
                        "text": ptext,
                        "embed_text": "",
                        "source_file": os.path.basename(path)
                    })

                    # level 3: gạch đầu dòng (nếu có). Nếu không có -> leaf là Điểm.
                    bullets = split_bullets(ptext)
                    if bullets:
                        for bi, bt in enumerate(bullets, 1):
                            emit_leaf(
                                items,
                                law=law_name, source_file=path,
                                chapter=chapter, section=section,
                                article_no=article_no, article_title=article_title,
                                clause_no=clause_no, point_letter=letter, bullet_idx=bi,
                                text=bt,
                                parents_ids={"point_id": point_id, "clause_id": clause_id, "article_id": article_id}
                            )
                    else:
                        emit_leaf(
                            items,
                            law=law_name, source_file=path,
                            chapter=chapter, section=section,
                            article_no=article_no, article_title=article_title,
                            clause_no=clause_no, point_letter=letter, bullet_idx=None,
                            text=ptext,
                            parents_ids={"clause_id": clause_id, "article_id": article_id}
                        )
            else:
                # Không có điểm -> leaf là Khoản (trừ preamble)
                if clause_no is not None:
                    emit_leaf(
                        items,
                        law=law_name, source_file=path,
                        chapter=chapter, section=section,
                        article_no=article_no, article_title=article_title,
                        clause_no=clause_no, point_letter=None, bullet_idx=None,
                        text=clause_body,
                        parents_ids={"article_id": article_id}
                    )
                else:
                    # preamble trước Khoản 1: thường ngắn, nhưng vẫn coi là leaf nếu cần
                    if clause_body.strip():
                        emit_leaf(
                            items,
                            law=law_name, source_file=path,
                            chapter=chapter, section=section,
                            article_no=article_no, article_title=article_title,
                            clause_no=None, point_letter=None, bullet_idx=None,
                            text=clause_body,
                            parents_ids={"article_id": article_id}
                        )

    out_path = OUT_DIR / (Path(path).stem + ".json")
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {law_name}: {len(items)} nodes/chunks → {out_path}")

# ===================== MAIN =====================
if __name__ == "__main__":
    print("🔄 Building lightweight hierarchical chunks (CPU friendly)…")
    for name, p in INPUTS:
        try:
            src = Path(p)
            if not src.exists():
                print(f"⚠️  Missing: {p}")
                continue
            raw = src.read_text(encoding="utf-8", errors="ignore")
            process_one(name, p)
        except Exception as e:
            print(f"❌ Lỗi xử lý {name} ({p}): {e}")
    print("✅ Done.")
