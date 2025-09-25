from __future__ import annotations
import os
import re
import io
import json
import time
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError, ExifTags
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

# ========================= CONFIG =========================
DATA_ROOT = Path("oral_pathology_data").resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = DATA_ROOT / "metadata.parquet"
CSV_PATH = DATA_ROOT / "metadata.csv"
TAGS_JSON = DATA_ROOT / ".tags.json"
HIDDEN_CASES_JSON = DATA_ROOT / ".hidden_cases.json"
SESSION_STATE_JSON = DATA_ROOT / ".session_state.json"
WORK_SESSION_JSON = DATA_ROOT / ".work_sessions.json"
USER_ACTIVITY_JSON = DATA_ROOT / ".user_activity.json"
CLINICAL_DIR = DATA_ROOT / "_clinical"
ensure_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)

APP_TITLE = "🦷 Oral Pathology Visual Explorer"

# UI defaults
MIN_IMG_SIZE_DEFAULT = 200
NEARBY_RATIO_DEFAULT = 0.25
SAVE_ALL_FALLBACK = True

# Work session settings
SESSION_TIMEOUT_MINUTES = 30
AUTO_SAVE_INTERVAL = 3

# ========================= ADVANCED SESSION STATE MANAGEMENT =========================
class WorkSessionManager:
    """Advanced work session management for oral pathology review"""
    
    def __init__(self):
        self.session_id = self._generate_session_id()
        self.start_time = time.time()
        self.last_activity = time.time()
        self.auto_save_counter = 0
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"oral_session_{timestamp}_{hash_part}"
    
    def update_activity(self):
        """Update last activity time"""
        self.last_activity = time.time()
    
    def is_session_expired(self) -> bool:
        """Check if session has expired"""
        return (time.time() - self.last_activity) > (SESSION_TIMEOUT_MINUTES * 60)
    
    def get_session_duration(self) -> str:
        """Get current session duration"""
        duration = time.time() - self.start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"
    
    def save_session_snapshot(self):
        """Save session snapshot for review tracking"""
        try:
            session_data = {
                "session_id": self.session_id,
                "start_time": self.start_time,
                "last_activity": self.last_activity,
                "duration_minutes": (time.time() - self.start_time) / 60,
                "edited_images": list(st.session_state.get("edited_images_set", set())),
                "images_processed": len(st.session_state.get("edited_images_set", set())),
                "timestamp": datetime.now().isoformat(),
                "session_type": "oral_pathology_review"
            }
            
            sessions = []
            if WORK_SESSION_JSON.exists():
                try:
                    sessions = json.loads(WORK_SESSION_JSON.read_text(encoding="utf-8"))
                except Exception:
                    pass
            
            sessions.append(session_data)
            sessions = sessions[-100:]  # Keep last 100 sessions
            
            WORK_SESSION_JSON.write_text(
                json.dumps(sessions, ensure_ascii=False, indent=2), 
                encoding="utf-8"
            )
        except Exception:
            pass

# ========================= SESSION STATE INITIALIZATION =========================
def init_session_state():
    """Initialize all session state variables with enhanced tracking"""
    defaults = {
        "selected_list": [],
        "custom_tags": {},
        "filter_tags": [],
        "hidden_cases": [],
        "lightbox_open": False,
        "lightbox_seq": [],
        "lightbox_idx": 0,
        "library_prefill": {},
        "pending_updates": {},
        "ui_update_flag": False,
        "last_metadata_update": 0,
        "page_state": {},
        "defer_rerun": False,
        "edited_images_set": set(),
        "last_visit_times": {},
        "edit_timestamps": {},
        "work_session_manager": WorkSessionManager(),
        "productivity_stats": {
            "images_edited_today": 0,
            "images_edited_session": 0,
            "total_edits": 0,
            "avg_time_per_edit": 0.0
        },
        "ui_preferences": {
            "show_edited_badge": True,
            "sort_edited_first": False,
            "auto_save_enabled": True,
            "show_productivity_stats": True,
            "cards_per_row": 3,
            "show_thumbnails": True
        },
        "quick_filters": {
            "show_only_unedited": False,
            "show_only_edited": False,
            "show_recent_edits": False
        },
        "view_mode": "grid",
        "image_size": "medium",
        "selection_keys": {},  # Track selection state without reruns
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    load_persistent_data()
    
    if hasattr(st.session_state, "work_session_manager"):
        st.session_state.work_session_manager.update_activity()

def load_persistent_data():
    """Load persistent data from disk"""
    # Load custom tags
    if TAGS_JSON.exists():
        try:
            st.session_state.custom_tags = json.loads(
                TAGS_JSON.read_text(encoding="utf-8")
            )
        except Exception:
            pass
    
    # Load hidden cases
    if HIDDEN_CASES_JSON.exists():
        try:
            st.session_state.hidden_cases = json.loads(
                HIDDEN_CASES_JSON.read_text(encoding="utf-8")
            )
        except Exception:
            pass
    
    # Load session state
    if SESSION_STATE_JSON.exists():
        try:
            session_data = json.loads(SESSION_STATE_JSON.read_text(encoding="utf-8"))
            
            if "edited_images_set" in session_data:
                st.session_state.edited_images_set = set(session_data["edited_images_set"])
            if "last_visit_times" in session_data:
                st.session_state.last_visit_times = session_data["last_visit_times"]
            if "edit_timestamps" in session_data:
                st.session_state.edit_timestamps = session_data["edit_timestamps"]
            if "productivity_stats" in session_data:
                st.session_state.productivity_stats.update(session_data["productivity_stats"])
            if "ui_preferences" in session_data:
                st.session_state.ui_preferences.update(session_data["ui_preferences"])
                
        except Exception:
            pass

def save_session_state():
    """Save enhanced session state to disk"""
    try:
        # Save tags
        TAGS_JSON.write_text(
            json.dumps(st.session_state.custom_tags, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # Save hidden cases
        HIDDEN_CASES_JSON.write_text(
            json.dumps(st.session_state.hidden_cases, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # Save session data
        session_data = {
            "edited_images_set": list(st.session_state.edited_images_set),
            "last_visit_times": st.session_state.last_visit_times,
            "edit_timestamps": st.session_state.edit_timestamps,
            "productivity_stats": st.session_state.productivity_stats,
            "ui_preferences": st.session_state.ui_preferences,
            "last_save": time.time()
        }
        
        SESSION_STATE_JSON.write_text(
            json.dumps(session_data, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        
        # Save work session snapshot periodically
        if hasattr(st.session_state, "work_session_manager"):
            st.session_state.work_session_manager.auto_save_counter += 1
            if st.session_state.work_session_manager.auto_save_counter >= AUTO_SAVE_INTERVAL:
                st.session_state.work_session_manager.save_session_snapshot()
                st.session_state.work_session_manager.auto_save_counter = 0
                
    except Exception:
        pass

def save_tags():
    """Save custom tags to disk"""
    try:
        TAGS_JSON.write_text(
            json.dumps(st.session_state.custom_tags, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass

# ========================= ENHANCED IMAGE EDIT TRACKING =========================
def mark_image_edited(image_path: str, edit_type: str = "manual"):
    """Mark an image as edited with detailed tracking"""
    current_time = time.time()
    
    st.session_state.edited_images_set.add(image_path)
    st.session_state.last_visit_times[image_path] = current_time
    
    if image_path not in st.session_state.edit_timestamps:
        st.session_state.edit_timestamps[image_path] = []
    
    st.session_state.edit_timestamps[image_path].append({
        "timestamp": current_time,
        "edit_type": edit_type,
        "session_id": st.session_state.work_session_manager.session_id
    })
    
    st.session_state.productivity_stats["images_edited_session"] += 1
    st.session_state.productivity_stats["total_edits"] += 1
    
    if st.session_state.ui_preferences["auto_save_enabled"]:
        save_session_state()

def get_image_edit_info(image_path: str) -> Dict[str, Any]:
    """Get detailed edit information for an image"""
    if image_path not in st.session_state.edited_images_set:
        return {"is_edited": False}
    
    edit_history = st.session_state.edit_timestamps.get(image_path, [])
    last_edit = edit_history[-1] if edit_history else {}
    
    return {
        "is_edited": True,
        "edit_count": len(edit_history),
        "last_edit_time": last_edit.get("timestamp", 0),
        "last_edit_type": last_edit.get("edit_type", "unknown"),
        "time_since_edit": time.time() - last_edit.get("timestamp", 0) if last_edit else 0
    }

def format_time_since_edit(seconds: float) -> str:
    """Format time since last edit in human readable format"""
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        return f"{int(seconds//60)} min ago"
    elif seconds < 86400:
        return f"{int(seconds//3600)} hours ago"
    else:
        return f"{int(seconds//86400)} days ago"

# ========================= ORAL PATHOLOGY KEYWORDS & SCORING =========================
ORAL_PATH_ANATOMY_KEYWORDS = {
    "lip": ["môi", "lip", "vermilion", "labial", "commissure", "mép", "môi trên", "môi dưới"],
    "cheek": ["má", "cheek", "buccal", "niêm mạc má", "mặt trong má"],
    "gingiva": ["nướu", "gingiva", "niêm mạc xương ổ răng", "lợi", "nướu dính", "nướu tự do"],
    "palate": ["khẩu cái", "palate", "khẩu cái cứng", "khẩu cái mềm", "hard palate", "soft palate", "vòm miệng"],
    "tongue": ["lưỡi", "tongue", "glossal", "lưng lưỡi", "bụng lưỡi", "đáy lưỡi", "dorsal", "ventral", "mặt lưỡi"],
    "floor_of_mouth": ["sàn miệng", "floor of mouth", "sàn lưỡi", "dưới lưỡi"],
    "jawbone": ["xương hàm", "jawbone", "mandible", "maxilla", "xương ổ răng", "hàm trên", "hàm dưới"],
    "other": ["khác", "other", "multiple sites", "nhiều vị trí", "toàn bộ"]
}

ORAL_PATH_LESION_TYPE_KEYWORDS = {
    "macule_patch": ["dát", "mảng", "macule", "patch", "nhiễm sắc", "pigmentation", "đốm"],
    "papule_nodule": ["sẩn", "hòn", "papule", "nodule", "cục", "u nhỏ"],
    "tumor": ["bướu", "u", "tumor", "neoplasm", "tăng sản", "hyperplasia", "sùi", "papilloma", "khối u"],
    "vesicle_bulla": ["mụn nước", "bóng nước", "vesicle", "bulla", "pemphigus", "pemphigoid", "phỏng nước"],
    "pustule": ["mụn mủ", "pustule", "áp xe", "abscess", "mủ"],
    "cyst": ["nang", "cyst", "u nang"],
    "ulcer_erosion": ["loét", "vết chợt", "ulcer", "erosion", "aphthous", "loét miệng", "loét tái phát"],
    "white_lesion": ["bạch sản", "leukoplakia", "lichen", "nấm", "candida", "mảng trắng", "tổn thương trắng"],
    "red_lesion": ["hồng sản", "erythroplakia", "viêm", "inflammation", "mảng đỏ", "tổn thương đỏ"],
    "mixed_lesion": ["tổn thương hỗn hợp", "mixed", "đỏ trắng", "erythroleukoplakia"],
    "fissure_groove": ["nứt nẻ", "chẻ", "rãnh", "fissure", "groove", "cleft"],
    "torus_exostosis": ["lồi xương", "torus", "exostosis", "đa lồi xương", "gai xương"]
}

ORAL_ANATOMY_OPTIONS = ["unknown"] + list(ORAL_PATH_ANATOMY_KEYWORDS.keys())
ORAL_LESION_OPTIONS = ["unknown"] + list(ORAL_PATH_LESION_TYPE_KEYWORDS.keys())

ORAL_PATH_SCORED_KEYWORDS = {
    "very_high": {
        "ung thư": 30, "carcinoma": 30, "sarcoma": 30, "ung thư tế bào gai": 25, 
        "squamous cell carcinoma": 25, "melanoma": 25, "lymphoma": 25,
        "bạch sản": 20, "leukoplakia": 20, "hồng sản": 22, "erythroplakia": 22,
        "dysplasia": 18, "loạn sản": 18,
        "pemphigus": 20, "lichen planus": 18, "lupus": 18, "pemphigoid": 20
    },
    "high": {
        "loét": 15, "ulcer": 15, "nang": 15, "cyst": 15, "bướu": 12, "u": 12, 
        "tumor": 12, "nấm candida": 15, "candidiasis": 15, "herpes": 12,
        "ranula": 14, "mucocele": 14, "fibroma": 12
    },
    "medium": {
        "tổn thương": 10, "lesion": 10, "viêm": 8, "inflammation": 8, 
        "tăng sản": 8, "hyperplasia": 8, "nhiễm sắc": 6, "pigmentation": 6,
        "mảng trắng": 10, "white patch": 10, "mảng đỏ": 10, "red patch": 10
    },
    "anatomy": {kw: 3 for kws in ORAL_PATH_ANATOMY_KEYWORDS.values() for kw in kws},
    "lesion_type": {kw: 3 for kws in ORAL_PATH_LESION_TYPE_KEYWORDS.values() for kw in kws},
    "low": {
        "sinh thiết": 4, "biopsy": 4, "lâm sàng": 3, "clinical": 3,
        "giải phẫu bệnh": 5, "histopathology": 5, "chẩn đoán": 3, "diagnosis": 3,
        "điều trị": 2, "treatment": 2
    },
    "negative": {
        "sơ đồ": -25, "diagram": -25, "illustration": -25, "schematic": -25,
        "khám": -5, "bệnh nhân": -5, "patient": -5
    }
}

# ========================= METADATA SCHEMA =========================
META_COLS = [
    "case_name", "image_path", "thumb_path", "page_num", "fig_num", "group_key",
    "caption", "context", "anatomical_site", "lesion_type", "confidence",
    "source", "saved_at", "bytes_md5", "relevance_score", "notes", "tags"
]

def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        return False

# ========================= UTILITY FUNCTIONS =========================
def safe_book_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\u00C0-\u1EF9]+", "_", str(name)).strip("_") or "unknown"

def unique_filename(p: Path) -> Path:
    if not p.exists():
        return p
    stem, ext, i = p.stem, p.suffix, 2
    while True:
        cand = p.with_name(f"{stem}-v{i}{ext}")
        if not cand.exists():
            return cand
        i += 1

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"-\s+\n?", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_chapter_tokens(s: str) -> str:
    if not s:
        return ""
    return re.sub(
        r"(?:Chapter|Chương)\s*\d+[:.\-\s]*", "", s, flags=re.IGNORECASE
    ).strip(" :.-\u00A0")

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def rect_min_distance(a: fitz.Rect, b: fitz.Rect) -> float:
    if a.intersects(b):
        return 0.0
    dx = max(b.x0 - a.x1, a.x0 - b.x1, 0)
    dy = max(b.y0 - a.y1, a.y0 - b.y1, 0)
    return (dx * dx + dy * dy) ** 0.5

def make_thumb(src: Path, dst: Path, max_side=512) -> Optional[Path]:
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            ensure_dir(dst.parent)
            im.save(dst, format="JPEG", quality=90, optimize=True)
        return dst
    except Exception:
        return None

def thumb_path_for(rel_img: str) -> Path:
    base = re.sub(r"[\\/]+", "__", rel_img)
    return DATA_ROOT / "_thumbs" / (Path(base).with_suffix(".jpg").name)

def highlight(text: str, query: str) -> str:
    if not query:
        return text
    try:
        pat = re.compile(re.escape(query), re.IGNORECASE)
        return pat.sub(lambda m: f"<mark style='background:#ffeb3b;'>{m.group(0)}</mark>", text)
    except Exception:
        return text

# ========================= OPTIMIZED METADATA OPERATIONS =========================
@st.cache_data(show_spinner="Loading metadata...", ttl=300)
def md_load_cached(last_update_timestamp: float = 0) -> pd.DataFrame:
    """Cached metadata loading"""
    path = PARQUET_PATH if _parquet_available() and PARQUET_PATH.exists() else CSV_PATH
    if not path.exists():
        return pd.DataFrame(columns=META_COLS)
    
    try:
        df = (pd.read_parquet(path) if _parquet_available() and path.suffix == ".parquet" 
              else pd.read_csv(path))
    except Exception as e:
        st.error(f"Error reading metadata: {e}")
        return pd.DataFrame(columns=META_COLS)
    
    for c in META_COLS:
        if c not in df.columns:
            df[c] = "" if c not in ["relevance_score"] else 0
        if 'confidence' in df.columns:
            df['confidence'] = df['confidence'].astype(str)
    return df

def md_load() -> pd.DataFrame:
    """Load metadata with caching"""
    return md_load_cached(st.session_state.last_metadata_update)

def md_save_immediate(df: pd.DataFrame) -> None:
    """Save metadata immediately"""
    for c in META_COLS:
        if c not in df.columns:
            df[c] = "" if c != "relevance_score" else 0
    
    df = df[META_COLS]
    use_parquet = _parquet_available()
    path = PARQUET_PATH if use_parquet else CSV_PATH
    backup_path = path.with_suffix(f"{path.suffix}.bak")
    
    try:
        if path.exists():
            shutil.copy2(path, backup_path)
        
        if use_parquet:
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        
        st.session_state.last_metadata_update = time.time()
        md_load_cached.clear()
        
    except Exception as e:
        st.error(f"Save failed: {e}. Backup available at: {backup_path}")

def md_update_by_paths_batch(updates_dict: Dict[str, Dict]) -> None:
    """Batch update multiple images"""
    df = md_load()
    
    for image_paths, updates in updates_dict.items():
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        mask = df["image_path"].isin(image_paths)
        for k, v in updates.items():
            if k in df.columns:
                df.loc[mask, k] = v
    
    md_save_immediate(df)

def md_delete_by_paths_batch(image_rels: List[str]) -> None:
    """Batch delete multiple images"""
    df = md_load()
    df = df[~df["image_path"].isin(image_rels)].copy()
    md_save_immediate(df)

def delete_duplicate_images():
    """Find and delete duplicate images, prioritizing edited ones"""
    df = md_load()
    if df.empty or 'bytes_md5' not in df.columns:
        return 0

    edited_set = st.session_state.edited_images_set
    
    md5_counts = df['bytes_md5'].value_counts()
    duplicate_md5s = md5_counts[md5_counts > 1].index.tolist()

    if not duplicate_md5s:
        return 0

    indices_to_delete = []
    
    for md5 in duplicate_md5s:
        group = df[df['bytes_md5'] == md5].copy()
        group['is_edited'] = group['image_path'].apply(lambda path: path in edited_set)
        group_sorted = group.sort_values(by=['is_edited', 'saved_at'], ascending=[False, False])
        record_to_keep = group_sorted.iloc[0]
        indices_to_delete.extend(group[group['image_path'] != record_to_keep['image_path']].index.tolist())

    if not indices_to_delete:
        return 0

    records_to_delete = df.loc[indices_to_delete]

    for _, row in records_to_delete.iterrows():
        try:
            rel_path = row['image_path']
            (DATA_ROOT / rel_path).unlink(missing_ok=True)
            
            thumb_path = row.get('thumb_path')
            if thumb_path and (DATA_ROOT / thumb_path).exists():
                (DATA_ROOT / thumb_path).unlink(missing_ok=True)
        except Exception as e:
            st.warning(f"Error deleting file {rel_path}: {e}")

    df_cleaned = df.drop(index=indices_to_delete)
    md_save_immediate(df_cleaned)

    paths_deleted = records_to_delete['image_path'].tolist()
    st.session_state.edited_images_set -= set(paths_deleted)
    
    return len(indices_to_delete)

# ========================= CAPTION & SCORING FUNCTIONS =========================
FIG_PATTERNS = [
    r"(?i)(?:Fig(?:ure)?|Hình|Ảnh)\s*([\dA-Za-z]{1,4}(?:\.\d+)?)\s*[.:\-]\s*(.+)",
    r"(?i)(?:Fig(?:ure)?|Hình|Ảnh)\s*[.:\-]\s*(.+)",
]

def extract_page_caption(page_text: str) -> Tuple[str, str]:
    for pat in FIG_PATTERNS:
        m = re.search(pat, page_text, flags=re.IGNORECASE)
        if m:
            gs = m.groups()
            fig = gs[0].strip() if len(gs) >= 2 else ""
            cap = strip_chapter_tokens(gs[-1].strip())
            return fig, cap
    return "", strip_chapter_tokens((page_text.split(".")[0] if page_text else "")[:200])

def get_image_rect(page: fitz.Page, xref: int) -> Optional[fitz.Rect]:
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return None
    
    for info in infos:
        if info.get("xref") == xref and "bbox" in info and len(info["bbox"]) == 4:
            return fitz.Rect(info["bbox"])
    return None

def nearby_text_for_image(page: fitz.Page, img_rect: Optional[fitz.Rect], nearby_ratio: float) -> str:
    text_all = normalize_text(page.get_text() or "")
    if not img_rect:
        return strip_chapter_tokens(text_all[:500])
    
    radius = max(15.0, page.rect.height * max(0.05, float(nearby_ratio)))
    blocks = page.get_text("blocks")
    bag = []
    
    for b in blocks:
        if len(b) < 5 or not isinstance(b[4], str):
            continue
        rect = fitz.Rect(b[:4])
        if rect.intersects(img_rect) or rect_min_distance(rect, img_rect) <= radius:
            bag.append(normalize_text(b[4]))
    
    ctx = " ".join(bag).strip()
    m = re.search(
        r"(?i)(?:Fig(?:ure)?|Hình|Ảnh)\s*[\dA-Za-z.\-]*\s*[.:\-]\s*(.+)", ctx
    )
    if m:
        return strip_chapter_tokens(m.group(1))[:400]
    
    return strip_chapter_tokens(ctx[:400]) or strip_chapter_tokens(text_all[:350])

def guess_labels(text: str) -> Tuple[str, str, str]:
    """Guess anatomical site and lesion type from text"""
    s = (text or "").lower()
    
    site_scores = {site: sum(2 for kw in kws if kw in s) 
                  for site, kws in ORAL_PATH_ANATOMY_KEYWORDS.items()}
    
    lesion_scores = {lt: sum(2 for kw in kws if kw in s) 
                    for lt, kws in ORAL_PATH_LESION_TYPE_KEYWORDS.items()}
    
    site = max(site_scores, key=site_scores.get) if any(site_scores.values()) else "unknown"
    lesion = max(lesion_scores, key=lesion_scores.get) if any(lesion_scores.values()) else "unknown"
    
    if site != "unknown" and lesion != "unknown":
        conf = "high"
    elif site != "unknown" or lesion != "unknown":
        conf = "medium"
    else:
        conf = "low"
    
    return site, lesion, conf

def calculate_relevance_score(text: str) -> int:
    """Calculate relevance score for oral pathology"""
    if not text:
        return 0
    
    t_lower = text.lower()
    score = 0
    found_groups = set()
    
    for group, kws in ORAL_PATH_SCORED_KEYWORDS.items():
        for kw, s in (kws.items() if isinstance(kws, dict) else []):
            if kw in t_lower:
                score += s
                found_groups.add(group)
    
    if ("anatomy" in found_groups or "lesion_type" in found_groups) and \
       ("very_high" in found_groups or "high" in found_groups):
        score += 20
    
    return max(0, score)

# ========================= PDF PROCESSING =========================
def process_pdf(uploaded, *, min_px: int, allow_duplicates: bool, save_all_if_no_kw: bool,
                nearby_ratio: float, relevance_threshold: int, progress=None) -> Tuple[List[str], int, Path]:
    """Extract and process images from PDF"""
    
    book = Path(uploaded.name).stem
    safe_b = safe_book_name(book)
    out_book_dir = DATA_ROOT / safe_b
    ensure_dir(out_book_dir)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    df = md_load()
    known = set(df.get("bytes_md5", [])) if "bytes_md5" in df.columns else set()
    saved_files: List[str] = []
    new_rows = []

    try:
        with fitz.open(tmp_path) as doc:
            total_pages = len(doc)
            for pno in range(total_pages):
                if progress:
                    progress((pno + 1) / max(1, total_pages), 
                            f"Processing page {pno+1}/{total_pages} — {book}")
                
                page = doc.load_page(pno)
                page_text = normalize_text(page.get_text() or "")
                page_fig, page_cap = extract_page_caption(page_text)
                images = page.get_images(full=True)
                
                if not images:
                    continue

                for idx, im in enumerate(images, start=1):
                    try:
                        xref = im[0] if im else None
                        if not isinstance(xref, int):
                            continue
                        
                        base = doc.extract_image(xref)
                        if not base or "image" not in base:
                            continue
                        
                        img_bytes: bytes = base["image"]
                        md5 = md5_bytes(img_bytes)
                        
                        if (not allow_duplicates) and md5 in known:
                            continue
                        
                        try:
                            with Image.open(io.BytesIO(img_bytes)) as im_pil:
                                w, h = im_pil.size
                            if max(w, h) < int(min_px):
                                continue
                        except Exception:
                            pass

                        rect = get_image_rect(page, xref)
                        near_cap = nearby_text_for_image(page, rect, nearby_ratio)
                        caption = near_cap if near_cap else page_cap
                        combined_text = f"{caption} {page_text}"
                        rel_score = calculate_relevance_score(combined_text)
                        
                        if (rel_score < relevance_threshold) and (not save_all_if_no_kw):
                            continue

                        m = re.search(
                            r"(?:Fig(?:ure)?|Hình|Ảnh)\s*([\dA-Za-z.]+)", 
                            near_cap or "", flags=re.IGNORECASE
                        )
                        fig_local = m.group(1) if m else ""
                        group_key = f"fig_{fig_local}" if fig_local else f"p{pno+1}"

                        page_folder = out_book_dir / f"p{pno+1}"
                        ensure_dir(page_folder)
                        ext = (base.get("ext") or "png").lower()
                        if ext not in ("png", "jpg", "jpeg"):
                            ext = "png"
                        
                        fname = f"{safe_b}_p{pno+1}_img{idx}.{ext}"
                        out_path = unique_filename(page_folder / fname)
                        out_path.write_bytes(img_bytes)

                        rel = str(out_path.relative_to(DATA_ROOT))
                        tpath = thumb_path_for(rel)
                        make_thumb(out_path, tpath, max_side=512)

                        site, lesion, conf = guess_labels(combined_text)
                        row = {
                            "case_name": book,
                            "image_path": rel,
                            "thumb_path": str(tpath.relative_to(DATA_ROOT)) if tpath and tpath.exists() else "",
                            "page_num": pno+1,
                            "fig_num": fig_local,
                            "group_key": group_key,
                            "caption": caption,
                            "context": (near_cap or page_text)[:700],
                            "anatomical_site": site,
                            "lesion_type": lesion,
                            "confidence": conf,
                            "source": "pdf",
                            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "bytes_md5": md5,
                            "relevance_score": int(rel_score),
                            "notes": "",
                            "tags": ""
                        }
                        new_rows.append(row)
                        saved_files.append(str(out_path))
                        known.add(md5)
                        
                    except Exception:
                        continue

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            md_save_immediate(df)
        
        return saved_files, len(saved_files), out_book_dir
        
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# ========================= WEB INGEST =========================
def ingest_web_html(url: str, *, min_px: int, allow_duplicates: bool, 
                   save_all_if_no_kw: bool, min_score: int) -> Tuple[int, int, Path]:
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        st.error(f"Failed to load page: {e}")
        return 0, 0, DATA_ROOT

    soup = BeautifulSoup(r.text, "html.parser")
    book = urlparse(url).netloc.replace("www.", "") or "web"
    out_book_dir = DATA_ROOT / safe_book_name(book) / "web"
    ensure_dir(out_book_dir)

    df = md_load()
    known = set(df.get("bytes_md5", [])) if "bytes_md5" in df.columns else set()
    kept = 0
    new_rows = []

    imgs = soup.find_all("img")
    for idx, img in enumerate(imgs, start=1):
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        
        img_url = urljoin(url, src)
        try:
            ir = requests.get(img_url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            if ir.status_code != 200 or not ir.content:
                continue
            
            md5 = md5_bytes(ir.content)
            if (not allow_duplicates) and md5 in known:
                continue

            try:
                with Image.open(io.BytesIO(ir.content)) as im:
                    w, h = im.size
                    fmt = (im.format or "JPEG").lower()
                if max(w, h) < int(min_px):
                    continue
            except UnidentifiedImageError:
                continue

            caption = normalize_text((img.get("alt") or img.get("title") or "")[:200])
            score = calculate_relevance_score(caption)
            if score < int(min_score) and not save_all_if_no_kw:
                continue

            ext = ".png" if fmt == "png" else ".jpg"
            fname = unique_filename(out_book_dir / f"{book}_web_img{idx}{ext}")
            fname.write_bytes(ir.content)

            rel = str(fname.relative_to(DATA_ROOT))
            tpath = thumb_path_for(rel)
            make_thumb(fname, tpath, max_side=512)

            site, lesion, conf = guess_labels(caption)
            row = {
                "case_name": book,
                "image_path": rel,
                "thumb_path": str(tpath.relative_to(DATA_ROOT)) if tpath and tpath.exists() else "",
                "page_num": 1,
                "fig_num": "",
                "group_key": "web",
                "caption": caption,
                "context": caption,
                "anatomical_site": site,
                "lesion_type": lesion,
                "confidence": conf,
                "source": f"web:{urlparse(url).netloc}",
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bytes_md5": md5,
                "relevance_score": int(score),
                "notes": "",
                "tags": ""
            }
            new_rows.append(row)
            kept += 1
            known.add(md5)
            
        except Exception:
            continue

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        md_save_immediate(df)
    
    return kept, kept, out_book_dir

# ========================= CLINICAL IMAGE UPLOAD =========================
def ingest_clinical_images(files: List, *, case_name: str, caption_prefix: str, 
                          site: str, lesion: str, tags: str, allow_duplicates: bool) -> int:
    if not files:
        return 0
    
    case = safe_book_name(case_name or "Clinical")
    out_dir = CLINICAL_DIR / case
    ensure_dir(out_dir)
    df = md_load()
    known = set(df.get("bytes_md5", [])) if "bytes_md5" in df.columns else set()
    added = 0
    new_rows = []
    
    for f in files:
        try:
            b = f.read()
            if not b:
                continue
            
            md5 = md5_bytes(b)
            if (not allow_duplicates) and md5 in known:
                continue
            
            with Image.open(io.BytesIO(b)) as im:
                w, h = im.size
                fmt = (im.format or "JPEG").lower()
            
            ext = ".png" if fmt == "png" else ".jpg"
            fname = unique_filename(out_dir / f"{case}_{int(time.time()*1000)}{ext}")
            fname.write_bytes(b)
            rel = str(fname.relative_to(DATA_ROOT))
            tpath = thumb_path_for(rel)
            make_thumb(fname, tpath, max_side=512)

            base_cap = Path(f.name).stem.replace("_", " ").replace("-", " ")
            cap = " ".join([x for x in [caption_prefix.strip(), base_cap] if x]).strip()
            ctx = f"clinical upload"
            combo_text = f"{cap} {ctx}"
            site_g, lesion_g, conf = guess_labels(combo_text)
            site_final = site if site != "(Auto)" else site_g
            lesion_final = lesion if lesion != "(Auto)" else lesion_g
            rel_score = calculate_relevance_score(combo_text)

            row = {
                "case_name": f"Clinical::{case}",
                "image_path": rel,
                "thumb_path": str(tpath.relative_to(DATA_ROOT)) if tpath and tpath.exists() else "",
                "page_num": 1,
                "fig_num": "",
                "group_key": "clinical",
                "caption": cap,
                "context": ctx,
                "anatomical_site": site_final,
                "lesion_type": lesion_final,
                "confidence": conf,
                "source": "clinical",
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bytes_md5": md5,
                "relevance_score": int(rel_score),
                "notes": "",
                "tags": tags.strip()
            }
            new_rows.append(row)
            known.add(md5)
            added += 1
            
        except Exception:
            continue
    
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        md_save_immediate(df)
    
    return added

# ========================= EXPORT FUNCTIONALITY =========================
def export_images(selected_paths: List[str]) -> Optional[bytes]:
    """Export selected images as ZIP"""
    import zipfile
    
    if not selected_paths:
        return None
    
    df = md_load()
    selected_df = df[df["image_path"].isin(selected_paths)]
    
    if selected_df.empty:
        return None
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add images
        for _, row in selected_df.iterrows():
            img_path = DATA_ROOT / row["image_path"]
            if img_path.exists():
                arcname = f"images/{Path(row['image_path']).name}"
                zf.write(img_path, arcname)
        
        # Add metadata CSV
        metadata_csv = selected_df.to_csv(index=False)
        zf.writestr("metadata.csv", metadata_csv)
        
        # Add summary JSON
        summary = {
            "export_date": datetime.now().isoformat(),
            "total_images": len(selected_df),
            "anatomical_sites": selected_df["anatomical_site"].value_counts().to_dict(),
            "lesion_types": selected_df["lesion_type"].value_counts().to_dict(),
            "confidence_levels": selected_df["confidence"].value_counts().to_dict(),
            "sources": selected_df["source"].value_counts().to_dict()
        }
        zf.writestr("summary.json", json.dumps(summary, indent=2))
    
    buffer.seek(0)
    return buffer.read()

def generate_analytics_report(df: pd.DataFrame) -> str:
    """Generate detailed analytics report"""
    report = []
    report.append("# 📊 Oral Pathology Analytics Report")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall Statistics
    report.append("## 📈 Overall Statistics")
    report.append(f"- **Total Images:** {len(df)}")
    report.append(f"- **Total Cases:** {df['case_name'].nunique()}")
    report.append(f"- **Average Relevance Score:** {df['relevance_score'].mean():.2f}")
    report.append("")
    
    # Confidence Distribution
    report.append("## 🎯 Confidence Distribution")
    conf_dist = df["confidence"].value_counts()
    for conf, count in conf_dist.items():
        percentage = (count / len(df)) * 100
        report.append(f"- **{conf.capitalize()}:** {count} ({percentage:.1f}%)")
    report.append("")
    
    # Anatomical Sites
    report.append("## 📍 Anatomical Sites Distribution")
    site_dist = df["anatomical_site"].value_counts().head(10)
    for site, count in site_dist.items():
        percentage = (count / len(df)) * 100
        report.append(f"- **{site}:** {count} ({percentage:.1f}%)")
    report.append("")
    
    # Lesion Types
    report.append("## 🔬 Lesion Types Distribution")
    lesion_dist = df["lesion_type"].value_counts().head(10)
    for lesion, count in lesion_dist.items():
        percentage = (count / len(df)) * 100
        report.append(f"- **{lesion}:** {count} ({percentage:.1f}%)")
    report.append("")
    
    # Source Distribution
    report.append("## 📁 Source Distribution")
    source_dist = df["source"].value_counts()
    for source, count in source_dist.items():
        percentage = (count / len(df)) * 100
        report.append(f"- **{source}:** {count} ({percentage:.1f}%)")
    report.append("")
    
    # Edit Progress
    edited_count = len(st.session_state.edited_images_set)
    unedited_count = len(df) - edited_count
    report.append("## ✏️ Edit Progress")
    report.append(f"- **Edited Images:** {edited_count}")
    report.append(f"- **Unedited Images:** {unedited_count}")
    report.append(f"- **Completion Rate:** {(edited_count / len(df) * 100) if len(df) > 0 else 0:.1f}%")
    report.append("")
    
    # Top Tags
    if "tags" in df.columns and df["tags"].notna().any():
        report.append("## 🏷️ Top Tags")
        all_tags = []
        for tags_str in df["tags"].dropna():
            all_tags.extend([t.strip() for t in tags_str.split(",") if t.strip()])
        
        from collections import Counter
        tag_counts = Counter(all_tags).most_common(10)
        for tag, count in tag_counts:
            report.append(f"  - **{tag}:** {count} uses")
    
    return "\n".join(report)

# ========================= ENHANCED UI COMPONENTS =========================
def render_productivity_dashboard():
    """Enhanced productivity dashboard with session management"""
    if not st.session_state.ui_preferences.get("show_productivity_stats", True):
        return
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Work Session Stats")
        
        session_duration = st.session_state.work_session_manager.get_session_duration()
        st.metric("Session Time", session_duration)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Edited (Session)", 
                st.session_state.productivity_stats["images_edited_session"],
                help="Images edited in current session"
            )
        
        with col2:
            total_edited = len(st.session_state.edited_images_set)
            st.metric("Total Edited", total_edited, help="Total images edited")
        
        if st.session_state.productivity_stats["avg_time_per_edit"] > 0:
            avg_time = st.session_state.productivity_stats["avg_time_per_edit"] / 60
            st.metric("Avg Time/Image", f"{avg_time:.1f} min")
        
        if st.button("💾 Save Session"):
            st.session_state.work_session_manager.save_session_snapshot()
            save_session_state()
            st.success("✅ Session saved!")

def render_tag_management_sidebar():
    """Render tag management in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🏷️ Tag Library Management")
        
        custom_tags = st.session_state.get("custom_tags", {})
        
        if custom_tags:
            st.metric("Custom Tags", len(custom_tags))
            
            if st.toggle("Show Tag Details"):
                for tag_name, tag_info in sorted(custom_tags.items()):
                    with st.expander(f"#{tag_name}"):
                        st.write(f"**Description:** {tag_info.get('description', 'No description')}")
                        st.write(f"**Category:** {tag_info.get('category', 'General')}")
                        st.write(f"**Usage:** {tag_info.get('usage_count', 0)} times")
                        st.write(f"**Created:** {tag_info.get('created_at', 'Unknown')[:10]}")
                        
                        if st.button(f"🗑️ Delete {tag_name}", key=f"del_tag_{tag_name}"):
                            del st.session_state.custom_tags[tag_name]
                            save_tags()
                            st.rerun()
        else:
            st.info("No custom tags yet. Create your first tag by editing an image!")

def render_notes_tags_form(row: pd.Series, rel: str, col_key: str):
    """Enhanced notes and tags form with custom tag management"""
    df = md_load()
    existing_notes = ""
    existing_tags = ""
    img_row = df[df["image_path"] == rel]
    
    if not img_row.empty:
        existing_notes = img_row.iloc[0].get("notes", "")
        existing_tags = img_row.iloc[0].get("tags", "")
    
    with st.form(f"notes_tags_form_{rel}_{col_key}"):
        user_notes = st.text_area(
            "📝 Notes:",
            value=existing_notes,
            height=100,
            placeholder="Add clinical observations, diagnosis notes, treatment plans..."
        )
        
        st.markdown("🏷️ **Tags Management:**")
        
        predefined_tags = ["malignant", "benign", "inflammatory", "traumatic", "infectious", 
                          "pre-malignant", "biopsy", "treatment", "follow-up", "differential"]
        
        custom_tag_library = st.session_state.get("custom_tags", {})
        current_tags = [t.strip() for t in existing_tags.split(",") if t.strip()]
        
        st.markdown("**Quick Tags:**")
        selected_quick_tags = []
        cols = st.columns(5)
        for i, tag in enumerate(predefined_tags):
            with cols[i % 5]:
                if st.checkbox(tag, value=tag in current_tags, key=f"qt_{rel}_{tag}_{col_key}"):
                    selected_quick_tags.append(tag)
        
        if custom_tag_library:
            st.markdown("**Custom Tag Library:**")
            selected_custom_tags = []
            custom_cols = st.columns(4)
            for i, (tag_name, tag_info) in enumerate(custom_tag_library.items()):
                with custom_cols[i % 4]:
                    tag_display = f"{tag_name}"
                    if tag_info.get('description'):
                        tag_display += f" ({tag_info['description'][:20]}...)"
                    
                    if st.checkbox(tag_display, value=tag_name in current_tags, 
                                  key=f"ct_{rel}_{tag_name}_{col_key}"):
                        selected_custom_tags.append(tag_name)
        else:
            selected_custom_tags = []
        
        st.markdown("---")
        st.markdown("**Add New Custom Tag:**")
        
        new_tag_col1, new_tag_col2 = st.columns([2, 3])
        with new_tag_col1:
            new_tag_name = st.text_input(
                "Tag name:", 
                placeholder="e.g., 'squamous-cell-ca'",
                key=f"new_tag_name_{rel}_{col_key}"
            )
        
        with new_tag_col2:
            new_tag_desc = st.text_input(
                "Description (optional):", 
                placeholder="e.g., 'Squamous cell carcinoma findings'",
                key=f"new_tag_desc_{rel}_{col_key}"
            )
        
        new_tag_category = st.selectbox(
            "Category:",
            ["General", "Pathology", "Treatment", "Location", "Severity", "Timing"],
            key=f"new_tag_cat_{rel}_{col_key}"
        )
        
        add_new_tag_col1, add_new_tag_col2 = st.columns([1, 1])
        
        add_to_library = add_new_tag_col1.checkbox(
            "Add to tag library", 
            value=True,
            key=f"add_to_lib_{rel}_{col_key}",
            help="Save this tag for future use across all images"
        )
        
        use_on_this_image = add_new_tag_col2.checkbox(
            "Apply to this image", 
            value=True,
            key=f"apply_new_{rel}_{col_key}"
        )
        
        st.markdown("**Manual Tags (comma separated):**")
        manual_tags = st.text_input(
            "One-time tags:",
            value="",
            placeholder="tag1, tag2, tag3",
            key=f"manual_tags_{rel}_{col_key}"
        )
        
        all_selected_tags = []
        all_selected_tags.extend(selected_quick_tags)
        all_selected_tags.extend(selected_custom_tags)
        
        if new_tag_name.strip() and use_on_this_image:
            all_selected_tags.append(new_tag_name.strip())
        
        if manual_tags.strip():
            manual_tag_list = [t.strip() for t in manual_tags.split(",") if t.strip()]
            all_selected_tags.extend(manual_tag_list)
        
        final_tags = ",".join(sorted(set(all_selected_tags)))
        
        if final_tags:
            st.markdown("**Tag Preview:**")
            tag_preview_html = " ".join([
                f"<span style='background:#e8eaf6; color:#3f51b5; padding:2px 6px; "
                f"border-radius:10px; font-size:11px; margin:2px;'>#{tag}</span>"
                for tag in sorted(set(all_selected_tags))
            ])
            st.markdown(tag_preview_html, unsafe_allow_html=True)
        
        if st.form_submit_button("💾 Save Notes & Tags", type="primary"):
            if new_tag_name.strip() and add_to_library:
                if "custom_tags" not in st.session_state:
                    st.session_state.custom_tags = {}
                
                st.session_state.custom_tags[new_tag_name.strip()] = {
                    "description": new_tag_desc.strip(),
                    "category": new_tag_category,
                    "created_at": datetime.now().isoformat(),
                    "usage_count": 1
                }
                
                save_tags()
                st.success(f"✅ Added '{new_tag_name}' to tag library!")
            
            for tag in selected_custom_tags:
                if tag in st.session_state.custom_tags:
                    st.session_state.custom_tags[tag]["usage_count"] = \
                        st.session_state.custom_tags[tag].get("usage_count", 0) + 1
            
            df = md_load()
            mask = df["image_path"] == rel
            df.loc[mask, "notes"] = user_notes
            df.loc[mask, "tags"] = final_tags
            md_save_immediate(df)
            
            if user_notes.strip() or final_tags.strip():
                mark_image_edited(rel, "notes_tags_added")
            
            st.success("✅ Saved notes and tags!")
            st.rerun()

def render_image_card_visual(row: pd.Series, sel_list: List[str], seq: List[str], 
                            query: str = "", col_key: str = "") -> None:
    """Enhanced visual image card with no-rerun selection"""
    rel = row["image_path"]
    img_abs = DATA_ROOT / rel
    if not img_abs.exists():
        return

    edit_info = get_image_edit_info(rel)
    is_edited = edit_info["is_edited"]
    
    conf = str(row.get("confidence") or "low").lower()
    border_color = {
        "high": "#4caf50",
        "medium": "#ff9800", 
        "low": "#f44336"
    }.get(conf, "#888")

    with st.container(border=True):
        tp = row.get("thumb_path", "")
        show_img = DATA_ROOT / tp if tp and (DATA_ROOT / tp).exists() else img_abs
        
        if st.button("🖼️", key=f"view_{rel}_{col_key}", help="View in lightbox"):
            st.session_state.lightbox_open = True
            st.session_state.lightbox_seq = seq
            st.session_state.lightbox_idx = seq.index(rel) if rel in seq else 0
            st.rerun()
        
        st.image(str(show_img), use_container_width=True)
        
        edit_badge = ""
        if st.session_state.ui_preferences.get("show_edited_badge", True):
            if is_edited:
                edit_count = edit_info["edit_count"]
                time_since = format_time_since_edit(edit_info["time_since_edit"])
                edit_badge = f" ✅ (x{edit_count}, {time_since})"
            else:
                edit_badge = " 📄 unedited"
        
        st.markdown(
            f"""<div style='background:linear-gradient(90deg, {border_color}22, transparent); 
            padding:6px; border-radius:4px; margin:4px 0;'>
            <b>📕 {row['case_name']}</b> • 📄 p{int(row['page_num'])} • 
            🏷️ {row.get('group_key') or '-'}{edit_badge}
            </div>""", 
            unsafe_allow_html=True
        )
        
        site = row.get("anatomical_site", "unknown")
        lesion = row.get("lesion_type", "unknown")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<span style='background:#e3f2fd; color:#1976d2; padding:2px 8px; "
                f"border-radius:12px; font-size:13px;'>📍 {site}</span>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<span style='background:#fce4ec; color:#c2185b; padding:2px 8px; "
                f"border-radius:12px; font-size:13px;'>🔬 {lesion}</span>",
                unsafe_allow_html=True
            )
        
        priority_indicator = ""
        if not is_edited:
            priority_indicator = " 🎯"
        
        st.markdown(
            f"""<div style='margin:6px 0;'>
            <span style='background:{border_color}22; padding:2px 6px; border-radius:4px; 
            border-left:3px solid {border_color};'>
            Confidence: <b>{conf}</b> • Score: <b>{int(row.get('relevance_score', 0))}</b>
            </span>{priority_indicator}</div>""",
            unsafe_allow_html=True
        )
        
        caption_text = highlight(row.get("caption", "")[:150], query)
        st.markdown(f"<i style='color:#555; font-size:14px;'>{caption_text}</i>", 
                   unsafe_allow_html=True)
        
        if row.get("tags"):
            tags = row.get("tags", "").split(",")
            tag_html = " ".join([
                f"<span style='background:#e8eaf6; color:#3f51b5; padding:1px 6px; "
                f"border-radius:10px; font-size:11px; margin:2px;'>#{t.strip()}</span>"
                for t in tags if t.strip()
            ])
            st.markdown(tag_html, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            is_selected = rel in sel_list
            selection_key = f"select_{rel}_{col_key}"
            
            if selection_key not in st.session_state.selection_keys:
                st.session_state.selection_keys[selection_key] = is_selected
            
            if st.checkbox("Select", key=selection_key, value=st.session_state.selection_keys[selection_key]):
                if rel not in st.session_state.selected_list:
                    st.session_state.selected_list.append(rel)
                    st.session_state.selection_keys[selection_key] = True
            else:
                if rel in st.session_state.selected_list:
                    st.session_state.selected_list.remove(rel)
                    st.session_state.selection_keys[selection_key] = False
        
        with c2:
            with st.popover("📝 Notes/Tags", use_container_width=True):
                render_notes_tags_form(row, rel, col_key)
        
        with c3:
            with st.popover("✏️ Edit", use_container_width=True):
                render_edit_controls_visual(row, rel, query, col_key)

def render_edit_controls_visual(row: pd.Series, rel: str, query: str, col_key: str):
    """Visual edit controls with improved UX and edit status warning"""
    edit_form_key = f"edit_form_{rel}_{col_key}_{hash(str(row))}"
    
    edit_info = get_image_edit_info(rel)
    
    if edit_info["is_edited"]:
        st.warning("⚠️ This image has been previously edited.")
        last_edit_time_str = format_time_since_edit(edit_info.get("time_since_edit", 0))
        st.write(f"Last edited: {last_edit_time_str}")
        st.caption("You can still make changes, but please ensure consistency.")
    
    with st.form(edit_form_key):
        new_caption = st.text_area("Caption", value=row.get("caption", ""), height=80)
        
        col1, col2 = st.columns(2)
        
        with col1:
            site_idx = (ORAL_ANATOMY_OPTIONS.index(row.get("anatomical_site", "unknown")) 
                       if row.get("anatomical_site") in ORAL_ANATOMY_OPTIONS else 0)
            sel_site = st.selectbox("📍 Anatomical Site", ORAL_ANATOMY_OPTIONS, index=site_idx)
            
            if sel_site == "other":
                custom_site = st.text_input("Specify custom site:")
                if custom_site:
                    sel_site = custom_site
        
        with col2:
            lesion_idx = (ORAL_LESION_OPTIONS.index(row.get("lesion_type", "unknown")) 
                         if row.get("lesion_type") in ORAL_LESION_OPTIONS else 0)
            sel_lesion = st.selectbox("🔬 Lesion Type", ORAL_LESION_OPTIONS, index=lesion_idx)
        
        new_page = st.number_input("Page number", min_value=1, value=int(row["page_num"]))
        new_group = st.text_input("Group key", value=row.get("group_key", ""))
        
        conf_options = ["low", "medium", "high"]
        current_conf = row.get("confidence", "low")
        new_conf = st.select_slider("Confidence", options=conf_options, 
                                   value=current_conf if current_conf in conf_options else "low")
        
        if st.toggle("👁️ Preview full image"):
            img_abs = DATA_ROOT / rel
            if img_abs.exists():
                st.image(str(img_abs), caption=f"Full image: {Path(rel).name}", 
                        use_container_width=True)
            
            st.markdown("**📋 Detailed Information:**")
            info_cols = st.columns(2)
            
            with info_cols[0]:
                st.write(f"**📕 Case:** {row.get('case_name', 'N/A')}")
                st.write(f"**📄 Page:** {int(row.get('page_num', 0))}")
                st.write(f"**🏷️ Group:** {row.get('group_key', 'N/A')}")
                st.write(f"**🎯 Confidence:** {row.get('confidence', 'unknown')}")
                
            with info_cols[1]:
                st.write(f"**📍 Anatomical Site:** {row.get('anatomical_site', 'unknown')}")
                st.write(f"**🔬 Lesion Type:** {row.get('lesion_type', 'unknown')}")
                st.write(f"**📊 Relevance Score:** {int(row.get('relevance_score', 0))}")
                st.write(f"**📅 Saved:** {row.get('saved_at', 'N/A')}")
            
            if row.get("context"):
                st.markdown("**📄 Context:**")
                ctx = highlight(row.get("context", ""), query)
                st.markdown(ctx, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        if col1.form_submit_button("💾 Save Changes", type="primary"):
            updates = {
                "caption": new_caption,
                "anatomical_site": sel_site,
                "lesion_type": sel_lesion,
                "group_key": new_group,
                "confidence": new_conf
            }
            
            if int(new_page) != int(row["page_num"]):
                updates["page_num"] = int(new_page)
            
            md_update_by_paths_batch({rel: updates})
            mark_image_edited(rel, "manual_edit")
            st.success("✅ Updated!")
            st.rerun()
        
        if col2.form_submit_button("🗑️ Delete Image", type="secondary"):
            (DATA_ROOT / rel).unlink(missing_ok=True)
            tpr = row.get("thumb_path", "")
            if tpr:
                (DATA_ROOT / tpr).unlink(missing_ok=True)
            
            df = md_load()
            df = df[df["image_path"] != rel]
            md_save_immediate(df)
            
            if rel in st.session_state.selected_list:
                st.session_state.selected_list.remove(rel)
            
            if rel in st.session_state.edited_images_set:
                st.session_state.edited_images_set.remove(rel)
            if rel in st.session_state.last_visit_times:
                del st.session_state.last_visit_times[rel]
            if rel in st.session_state.edit_timestamps:
                del st.session_state.edit_timestamps[rel]
            
            st.success("✅ Deleted!")
            st.rerun()

def apply_enhanced_sorting(df: pd.DataFrame) -> pd.DataFrame:
    """Apply enhanced sorting with edit status priority"""
    if df.empty:
        return df
    
    df = df.copy()
    df["is_edited"] = df["image_path"].apply(lambda x: x in st.session_state.edited_images_set)
    df["last_visit"] = df["image_path"].apply(lambda x: st.session_state.last_visit_times.get(x, 0))
    
    quick_filters = st.session_state.quick_filters
    if quick_filters.get("show_only_unedited"):
        df = df[~df["is_edited"]]
    elif quick_filters.get("show_only_edited"):
        df = df[df["is_edited"]]
    elif quick_filters.get("show_recent_edits"):
        recent_threshold = time.time() - (24 * 3600)
        df = df[df["last_visit"] > recent_threshold]
    
    if st.session_state.ui_preferences.get("sort_edited_first", False):
        df = df.sort_values(
            ["is_edited", "last_visit"], 
            ascending=[False, False]
        )
    else:
        df = df.sort_values(
            ["is_edited", "last_visit"], 
            ascending=[True, False]
        )
    
    return df.drop(columns=["is_edited", "last_visit"])

def render_lightbox_enhanced():
    """Enhanced lightbox with edit capabilities"""
    if not (st.session_state.get("lightbox_open") and st.session_state.get("lightbox_seq")):
        return
    
    seq = st.session_state.lightbox_seq
    idx = st.session_state.lightbox_idx
    idx = max(0, min(idx, len(seq) - 1))
    rel = seq[idx]
    p = DATA_ROOT / rel
    
    if not p.exists():
        st.session_state.lightbox_open = False
        return
    
    st.markdown("<div class='lightbox'><div class='lightbox-inner'>", unsafe_allow_html=True)
    st.image(str(p), use_container_width=True)
    
    df = md_load()
    row = df[df["image_path"] == rel]
    
    if not row.empty:
        r = row.iloc[0]
        edit_info = get_image_edit_info(rel)
        
        cap = r["caption"]
        edit_status = "✅ Edited" if edit_info["is_edited"] else "📄 Unedited"
        
        st.markdown(f"<div class='lightbox-caption'><strong>{cap}</strong><br>"
                   f"<small>{r['case_name']} • p{r['page_num']} • {edit_status}</small></div>", 
                   unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns([1, 3, 2, 1])
    
    with c1:
        if st.button("⟵ Prev", key="lb_prev"):
            st.session_state.lightbox_idx = (idx - 1) % len(seq)
    
    with c2:
        if st.button("✖ Close", use_container_width=True, key="lb_close"):
            st.session_state.lightbox_open = False
    
    with c3:
        if not row.empty and not edit_info["is_edited"]:
            if st.button("⚡ Quick Edit", use_container_width=True, key="lb_quick_edit"):
                mark_image_edited(rel, "lightbox_quick")
                st.success("Marked as edited!")
                st.rerun()
    
    with c4:
        if st.button("Next ⟶", key="lb_next"):
            st.session_state.lightbox_idx = (idx + 1) % len(seq)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# ========================= MAIN APPLICATION =========================
def main():
    st.set_page_config(
        page_title=APP_TITLE, 
        page_icon="🦷", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    :root { 
        --primary: #2196f3;
        --secondary: #ff9800;
        --success: #4caf50;
        --danger: #f44336;
        --dark: #263238;
        --light: #f5f5f5;
        --border-radius: 8px;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    h1, h2, h3 {
        color: var(--dark);
        font-weight: 600;
    }
    
    .stButton button {
        border-radius: var(--border-radius);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .tag-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
        background: linear-gradient(135deg, var(--primary)22, var(--secondary)22);
    }
    
    .confidence-high { border-left: 4px solid var(--success); }
    .confidence-medium { border-left: 4px solid var(--secondary); }
    .confidence-low { border-left: 4px solid var(--danger); }
    
    .image-card {
        transition: transform 0.2s ease;
    }
    
    .image-card:hover {
        transform: scale(1.02);
    }
    
    .selection-strip {
        position: sticky;
        top: 0;
        z-index: 100;
        background: white;
        border-radius: var(--border-radius);
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 16px;
    }
    
    mark {
        background: #ffeb3b;
        padding: 1px 3px;
        border-radius: 3px;
    }
    
    .lightbox{
        position:fixed;left:0;top:0;width:100vw;height:100vh;
        background:rgba(0,0,0,.8);z-index:1000;display:flex;
        align-items:center;justify-content:center;
    }
    .lightbox-inner{
        width:min(96vw,1200px); background:#111; padding:8px; border-radius:8px;
    }
    .lightbox-caption{
        color:#ddd; font-size:14px; margin-top:8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    if st.session_state.ui_preferences.get("auto_save_enabled", True):
        save_session_state()
    
    render_productivity_dashboard()
    render_tag_management_sidebar()
    
    st.markdown("""
    <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
    font-size: 2.5rem; text-align: center; margin-bottom: 2rem;'>
    🦷 Oral Pathology Visual Explorer
    </h1>
    """, unsafe_allow_html=True)
    
    df = md_load()
    if not df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Images", len(df), delta=None)
        with col2:
            high_conf = len(df[df["confidence"] == "high"])
            st.metric("High Confidence", high_conf, delta=None)
        with col3:
            unique_sites = df["anatomical_site"].nunique()
            st.metric("Anatomical Sites", unique_sites)
        with col4:
            unique_lesions = df["lesion_type"].nunique()
            st.metric("Lesion Types", unique_lesions)
        with col5:
            avg_score = df["relevance_score"].mean()
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        total_images = len(df)
        edited_images = len(st.session_state.edited_images_set)
        completion_rate = (edited_images / total_images * 100) if total_images > 0 else 0
        
        st.progress(min(completion_rate / 100, 1.0), f"Review Progress: {edited_images}/{total_images} ({completion_rate:.1f}%)")
    
    tab_extract, tab_clinical, tab_library, tab_analytics, tab_settings = st.tabs([
        "📥 Extract from PDF/Web", "🏥 Clinical Upload", 
        "🖼️ Visual Library", "📊 Analytics", "⚙️ Advanced Settings"
    ])
    
    with tab_extract:
        extract_col1, extract_col2 = st.columns(2)
        
        with extract_col1:
            st.subheader("📄 PDF Extraction")
            up = st.file_uploader("Choose PDF file", type=["pdf"])
            
            with st.expander("⚙️ PDF Settings", expanded=False):
                min_px = st.slider("Min image size (px)", 100, 500, MIN_IMG_SIZE_DEFAULT)
                allow_dup = st.checkbox("Allow duplicates", value=False)
                save_all = st.checkbox("Save all images", value=SAVE_ALL_FALLBACK)
                nearby_ratio = st.slider("Caption search radius", 0.05, 0.40, NEARBY_RATIO_DEFAULT)
                min_score_pdf = st.slider("Min relevance score", 0, 50, 10)
            
            if st.button("🚀 Extract from PDF", type="primary", disabled=up is None):
                if up and up.size <= 200 * 1024 * 1024:
                    pb = st.progress(0.0)
                    with st.spinner("Extracting..."):
                        paths, n, book_dir = process_pdf(
                            up, min_px=min_px, allow_duplicates=allow_dup,
                            save_all_if_no_kw=save_all, nearby_ratio=nearby_ratio,
                            relevance_threshold=min_score_pdf,
                            progress=lambda p, m: pb.progress(p)
                        )
                    pb.progress(1.0)
                    
                    if n > 0:
                        st.success(f"✅ Extracted {n} images to: {book_dir}")
                    else:
                        st.warning("No relevant images found. Try adjusting filters.")
                else:
                    st.error("File exceeds 200MB limit.")
        
        with extract_col2:
            st.subheader("🌐 Web Extraction")
            url = st.text_input("Enter URL", placeholder="https://example.com/oral-pathology")
            
            with st.expander("⚙️ Web Settings", expanded=False):
                min_px_w = st.slider("Min image size (px)", 100, 500, MIN_IMG_SIZE_DEFAULT, key="web_min")
                allow_dup_w = st.checkbox("Allow duplicates", value=False, key="web_dup")
                save_all_w = st.checkbox("Save all images", value=True, key="web_all")
                min_score_web = st.slider("Min relevance score", 0, 50, 5, key="web_score")
            
            if st.button("🌐 Extract from Web", type="primary", disabled=not url):
                with st.spinner(f"Extracting from {url}..."):
                    kept, _, out_dir = ingest_web_html(
                        url.strip(), min_px=min_px_w, allow_duplicates=allow_dup_w,
                        save_all_if_no_kw=save_all_w, min_score=min_score_web
                    )
                
                if kept > 0:
                    st.success(f"✅ Saved {kept} images to: {out_dir}")
                else:
                    st.warning("No images found. Check URL or adjust settings.")

    with tab_clinical:
        st.subheader("🏥 Clinical Image Upload")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            files = st.file_uploader(
                "Select clinical images",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True
            )
            
            c1, c2 = st.columns(2)
            with c1:
                case_name = st.text_input("Case name", value="Case001")
                caption_prefix = st.text_input("Caption prefix", placeholder="Pre-op view of...")
            
            with c2:
                site_options = ["(Auto)"] + ORAL_ANATOMY_OPTIONS
                site_pick = st.selectbox("Anatomical site", site_options)
                
                if site_pick == "other":
                    custom_site = st.text_input("Custom site:")
                    if custom_site:
                        site_pick = custom_site
                
                lesion_pick = st.selectbox("Lesion type", ["(Auto)"] + ORAL_LESION_OPTIONS)
            
            st.markdown("### 🏷️ Tag Management")
            
            clinical_tags = ["pre-op", "post-op", "intra-op", "follow-up", 
                           "1-week", "1-month", "3-months", "6-months", "1-year"]
            
            selected_clinical_tags = st.multiselect(
                "Select clinical tags:",
                clinical_tags,
                default=[]
            )
            
            manual_clinical_tags = st.text_input(
                "Additional tags (comma separated):",
                placeholder="tag1, tag2, tag3"
            )
            
            all_clinical_tags = selected_clinical_tags.copy()
            if manual_clinical_tags:
                all_clinical_tags.extend([t.strip() for t in manual_clinical_tags.split(",") if t.strip()])
            
            final_clinical_tags = ",".join(all_clinical_tags)
            
            allow_dup_clinical = st.checkbox("Allow duplicate images", value=False, key="clinical_dup")
            
            if st.button("📤 Upload Clinical Images", type="primary", disabled=not files):
                with st.spinner(f"Uploading {len(files)} images..."):
                    added = ingest_clinical_images(
                        files, 
                        case_name=case_name, 
                        caption_prefix=caption_prefix,
                        site=site_pick, 
                        lesion=lesion_pick, 
                        tags=final_clinical_tags,
                        allow_duplicates=allow_dup_clinical
                    )
                
                if added > 0:
                    st.success(f"✅ Successfully uploaded {added} images to case: {case_name}")
                else:
                    st.warning("No new images added. They might be duplicates.")
        
        with col2:
            st.info("""
            **📋 Clinical Upload Guidelines:**
            
            - Use meaningful case names
            - Add descriptive captions
            - Select anatomical sites
            - Tag temporal information
            - Keep image quality high
            - Avoid patient identifiers
            """)
            
            if files:
                st.write(f"**Selected:** {len(files)} files")
                total_size = sum(f.size for f in files) / (1024 * 1024)
                st.write(f"**Total Size:** {total_size:.2f} MB")

    with tab_library:
        df = md_load()
        
        if df.empty:
            st.info("📭 No images yet. Start by extracting from PDFs, web pages, or uploading clinical images.")
        else:
            # Quick Filters
            st.markdown("### 🔍 Quick Filters")
            quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
            
            with quick_col1:
                if st.button("📄 Show Unedited Only"):
                    st.session_state.quick_filters["show_only_unedited"] = True
                    st.session_state.quick_filters["show_only_edited"] = False
                    st.session_state.quick_filters["show_recent_edits"] = False
                    st.rerun()
            
            with quick_col2:
                if st.button("✅ Show Edited Only"):
                    st.session_state.quick_filters["show_only_edited"] = True
                    st.session_state.quick_filters["show_only_unedited"] = False
                    st.session_state.quick_filters["show_recent_edits"] = False
                    st.rerun()
            
            with quick_col3:
                if st.button("🕐 Recent Edits (24h)"):
                    st.session_state.quick_filters["show_recent_edits"] = True
                    st.session_state.quick_filters["show_only_unedited"] = False
                    st.session_state.quick_filters["show_only_edited"] = False
                    st.rerun()
            
            with quick_col4:
                if st.button("🔄 Clear Filters"):
                    st.session_state.quick_filters = {
                        "show_only_unedited": False,
                        "show_only_edited": False,
                        "show_recent_edits": False
                    }
                    st.rerun()
            
            # Search and Filters
            with st.container(border=True):
                search_col1, search_col2 = st.columns([3, 1])
                
                with search_col1:
                    search_query = st.text_input(
                        "🔍 Search images",
                        placeholder="Search in captions, context, tags...",
                        key="search_query"
                    )
                
                with search_col2:
                    sort_option = st.selectbox(
                        "Sort by",
                        ["Relevance Score", "Confidence", "Date Added", "Edit Status"],
                        key="sort_option"
                    )
            
            # Advanced Filters
            with st.expander("🔧 Advanced Filters", expanded=False):
                filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                
                with filter_col1:
                    filter_cases = st.multiselect(
                        "Cases",
                        sorted(df["case_name"].unique()),
                        key="filter_cases"
                    )
                
                with filter_col2:
                    filter_sites = st.multiselect(
                        "Anatomical Sites",
                        sorted([s for s in df["anatomical_site"].unique() if s]),
                        key="filter_sites"
                    )
                
                with filter_col3:
                    filter_lesions = st.multiselect(
                        "Lesion Types",
                        sorted([l for l in df["lesion_type"].unique() if l]),
                        key="filter_lesions"
                    )
                
                with filter_col4:
                    filter_confidence = st.multiselect(
                        "Confidence",
                        ["high", "medium", "low"],
                        key="filter_confidence"
                    )
                
                score_range = st.slider(
                    "Relevance Score Range",
                    min_value=int(df["relevance_score"].min()),
                    max_value=int(df["relevance_score"].max()),
                    value=(int(df["relevance_score"].min()), int(df["relevance_score"].max())),
                    key="score_range"
                )
            
            # Apply filters
            filtered_df = df.copy()
            
            if search_query:
                mask = (
                    filtered_df["caption"].str.contains(search_query, case=False, na=False) |
                    filtered_df["context"].str.contains(search_query, case=False, na=False) |
                    filtered_df["tags"].str.contains(search_query, case=False, na=False) |
                    filtered_df["notes"].str.contains(search_query, case=False, na=False)
                )
                filtered_df = filtered_df[mask]
            
            if filter_cases:
                filtered_df = filtered_df[filtered_df["case_name"].isin(filter_cases)]
            
            if filter_sites:
                filtered_df = filtered_df[filtered_df["anatomical_site"].isin(filter_sites)]
            
            if filter_lesions:
                filtered_df = filtered_df[filtered_df["lesion_type"].isin(filter_lesions)]
            
            if filter_confidence:
                filtered_df = filtered_df[filtered_df["confidence"].isin(filter_confidence)]
            
            score_min, score_max = score_range
            filtered_df = filtered_df[
                (filtered_df["relevance_score"] >= score_min) & 
                (filtered_df["relevance_score"] <= score_max)
            ]
            
            # Apply sorting
            if sort_option == "Relevance Score":
                filtered_df = filtered_df.sort_values("relevance_score", ascending=False)
            elif sort_option == "Confidence":
                conf_order = {"high": 0, "medium": 1, "low": 2}
                filtered_df["conf_order"] = filtered_df["confidence"].map(conf_order).fillna(3)
                filtered_df = filtered_df.sort_values("conf_order").drop(columns=["conf_order"])
            elif sort_option == "Date Added":
                filtered_df = filtered_df.sort_values("saved_at", ascending=False)
            elif sort_option == "Edit Status":
                filtered_df = apply_enhanced_sorting(filtered_df)
            else:
                filtered_df = apply_enhanced_sorting(filtered_df)
            
            # Selection strip
            selected = st.session_state.selected_list
            if selected:
                with st.container():
                    st.markdown(
                        f"""<div class='selection-strip'>
                        <b>📌 Selected: {len(selected)} images</b>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    
                    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                    
                    with action_col1:
                        if st.button("✅ Mark All as Edited"):
                            for path in selected:
                                mark_image_edited(path, "batch_mark")
                            st.success(f"✅ Marked {len(selected)} images as edited!")
                            st.rerun()
                    
                    with action_col2:
                        if st.button("📥 Export Selected"):
                            export_data = export_images(selected)
                            if export_data:
                                st.download_button(
                                    "💾 Download ZIP",
                                    export_data,
                                    f"oral_pathology_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    "application/zip"
                                )
                    
                    with action_col3:
                        if st.button("🗑️ Delete Selected"):
                            if st.checkbox("Confirm deletion", key="confirm_delete"):
                                for path in selected:
                                    img_path = DATA_ROOT / path
                                    img_path.unlink(missing_ok=True)
                                    
                                    thumb_rel = thumb_path_for(path)
                                    thumb_rel.unlink(missing_ok=True)
                                
                                md_delete_by_paths_batch(selected)
                                st.session_state.selected_list = []
                                st.success(f"✅ Deleted {len(selected)} images!")
                                st.rerun()
                    
                    with action_col4:
                        if st.button("❌ Clear Selection"):
                            st.session_state.selected_list = []
                            st.session_state.selection_keys = {}
                            st.rerun()
            
            # Display settings
            st.markdown("### 📐 Display Settings")
            display_col1, display_col2, display_col3 = st.columns(3)
            
            with display_col1:
                cards_per_row = st.selectbox(
                    "Cards per row",
                    [2, 3, 4, 5],
                    index=1,
                    key="cards_per_row"
                )
            
            with display_col2:
                items_per_page = st.selectbox(
                    "Items per page",
                    [12, 24, 36, 48, 60],
                    index=1,
                    key="items_per_page"
                )
            
            with display_col3:
                if st.checkbox("Sort edited first", value=st.session_state.ui_preferences.get("sort_edited_first", False)):
                    st.session_state.ui_preferences["sort_edited_first"] = True
                    save_session_state()
                    st.rerun()
                else:
                    st.session_state.ui_preferences["sort_edited_first"] = False
                    save_session_state()
            
            # Pagination
            total_items = len(filtered_df)
            total_pages = (total_items + items_per_page - 1) // items_per_page
            
            if total_items > 0:
                page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
                
                with page_col2:
                    current_page = st.number_input(
                        f"Page (1-{total_pages})",
                        min_value=1,
                        max_value=max(1, total_pages),
                        value=1,
                        key="current_page"
                    )
                
                start_idx = (current_page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                
                st.info(f"Showing {start_idx + 1}-{end_idx} of {total_items} images")
                
                # Display images in grid
                page_df = filtered_df.iloc[start_idx:end_idx]
                seq_list = page_df["image_path"].tolist()
                
                cols = st.columns(cards_per_row)
                for idx, (_, row) in enumerate(page_df.iterrows()):
                    with cols[idx % cards_per_row]:
                        render_image_card_visual(row, selected, seq_list, search_query, f"lib_{idx}")
            else:
                st.warning("No images match your filters.")
            
            render_lightbox_enhanced()

    with tab_analytics:
        df = md_load()
        
        if df.empty:
            st.info("📊 No data available for analytics. Start by adding some images.")
        else:
            st.markdown("## 📊 Oral Pathology Analytics Dashboard")
            
            # Summary metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                total_images = len(df)
                st.metric("Total Images", total_images)
            
            with metric_col2:
                unique_cases = df["case_name"].nunique()
                st.metric("Total Cases", unique_cases)
            
            with metric_col3:
                avg_relevance = df["relevance_score"].mean()
                st.metric("Avg Relevance Score", f"{avg_relevance:.1f}")
            
            with metric_col4:
                high_conf_pct = (df["confidence"] == "high").mean() * 100
                st.metric("High Confidence %", f"{high_conf_pct:.1f}%")
            
            # Additional metrics
            metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
            
            with metric_col5:
                clinical_count = df[df["source"].str.startswith("clinical", na=False)].shape[0]
                st.metric("Clinical Images", clinical_count)
            
            with metric_col6:
                pdf_count = df[df["source"] == "pdf"].shape[0]
                st.metric("PDF Images", pdf_count)
            
            with metric_col7:
                edited_count = len(st.session_state.edited_images_set)
                st.metric("Edited Images", edited_count)
            
            with metric_col8:
                completion_rate = (edited_count / len(df) * 100) if len(df) > 0 else 0
                st.metric("Review Progress", f"{completion_rate:.1f}%")
            
            st.markdown("---")
            
            # Charts and visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Distribution by Anatomical Site")
                site_dist = df["anatomical_site"].value_counts()
                st.bar_chart(site_dist)
            
            with col2:
                st.markdown("### Distribution by Lesion Type")
                lesion_dist = df["lesion_type"].value_counts()
                st.bar_chart(lesion_dist)
            
            st.markdown("### Confidence Distribution")
            conf_dist = df["confidence"].value_counts()
            st.bar_chart(conf_dist)
            
            st.markdown("### Relevance Score Distribution")
            score_hist = df["relevance_score"].value_counts(bins=10).sort_index()
            st.line_chart(score_hist)
            
            st.markdown("---")
            st.markdown("### 📥 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Export to CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"oral_pathology_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Export Metadata"):
                    metadata_json = df.to_json(orient="records", indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=metadata_json,
                        file_name=f"metadata_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("Generate Report"):
                    report = generate_analytics_report(df)
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )

    with tab_settings:
        st.subheader("⚙️ Advanced Settings & Data Management")
        
        st.markdown("### 🧹 Data Maintenance")
        c1, c2, c3, c4 = st.columns(4)
        
        if c1.button("🧹 Clear metadata (reset)", type="secondary"):
            with st.spinner("Clearing metadata..."):
                PARQUET_PATH.unlink(missing_ok=True)
                CSV_PATH.unlink(missing_ok=True)
                PARQUET_PATH.with_suffix(".parquet.bak").unlink(missing_ok=True)
                CSV_PATH.with_suffix(".csv.bak").unlink(missing_ok=True)
                md_load_cached.clear()
            st.success("✅ Metadata cleared.")
        
        if c2.button("🧱 Rebuild thumbnails"):
            with st.spinner("Rebuilding thumbnails..."):
                df = md_load()
                ok = 0
                for i, r in df.iterrows():
                    p = DATA_ROOT / str(r["image_path"])
                    if not p.exists():
                        continue
                    t = thumb_path_for(str(r["image_path"]))
                    if make_thumb(p, t):
                        ok += 1
                    df.loc[i, "thumb_path"] = str(t.relative_to(DATA_ROOT)) if t.exists() else ""
                md_save_immediate(df)
            st.success(f"✅ Created/updated {ok} thumbnails.")
        
        if c3.button("🧰 Data Health Check"):
            with st.spinner("Checking data integrity..."):
                df = md_load()
                missing_files = df[~df["image_path"].apply(lambda p: (DATA_ROOT / str(p)).exists())]
                
                all_files = []
                for root, _, files in os.walk(DATA_ROOT):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            try:
                                all_files.append(str(Path(root, f).relative_to(DATA_ROOT)))
                            except Exception:
                                pass
                
                orphan = sorted(set(all_files) - set(df["image_path"].astype(str)))
            
            st.warning(f"Missing files: {len(missing_files)} • Orphan files: {len(orphan)}")
            
            if len(missing_files) > 0:
                st.write("Missing images (have metadata, missing files):")
                st.dataframe(missing_files[["case_name", "image_path", "caption"]])
            
            if len(orphan) > 0:
                st.write("Orphan files (have files, no metadata):")
                st.code("\n".join(orphan[:100]))
        
        if c4.button("🔧 Remove Duplicates"):
            with st.spinner("Finding and removing duplicates..."):
                deleted_count = delete_duplicate_images()
            
            if deleted_count > 0:
                st.success(f"✅ Removed {deleted_count} duplicate images.")
            else:
                st.info("No duplicates found.")

        st.markdown("---")
        st.markdown("### 📊 Work Session Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Session", st.session_state.work_session_manager.get_session_duration())
            st.metric("Images Edited (Session)", st.session_state.productivity_stats["images_edited_session"])
            
            if st.button("💾 Save Work Session"):
                st.session_state.work_session_manager.save_session_snapshot()
                save_session_state()
                st.success("✅ Work session saved!")
        
        with col2:
            st.metric("Total Edited Images", len(st.session_state.edited_images_set))
            
            if st.button("🔄 Reset Edit Tracking", type="secondary"):
                if st.button("⚠️ Confirm Reset", key="confirm_reset_tracking"):
                    st.session_state.edited_images_set = set()
                    st.session_state.last_visit_times = {}
                    st.session_state.edit_timestamps = {}
                    st.session_state.productivity_stats = {
                        "images_edited_today": 0,
                        "images_edited_session": 0,
                        "total_edits": 0,
                        "avg_time_per_edit": 0.0
                    }
                    save_session_state()
                    st.success("✅ Edit tracking reset!")
                    st.rerun()
        
        with col3:
            if WORK_SESSION_JSON.exists():
                try:
                    sessions = json.loads(WORK_SESSION_JSON.read_text(encoding="utf-8"))
                    if sessions:
                        last_session = sessions[-1]
                        st.metric(
                            "Last Session", 
                            f"{int(last_session.get('duration_minutes', 0))} min",
                            f"{last_session.get('images_processed', 0)} images"
                        )
                        
                        if st.button("📈 View Session History"):
                            st.subheader("Work Session History")
                            df_sessions = pd.DataFrame(sessions[-20:])
                            df_sessions['duration_hours'] = df_sessions['duration_minutes'] / 60
                            df_sessions['timestamp'] = pd.to_datetime(df_sessions['timestamp'])
                            st.dataframe(df_sessions[['timestamp', 'duration_hours', 'images_processed']])
                except Exception:
                    pass

        st.markdown("---")
        st.markdown("### 📚 Case/Book Management")
        
        df = md_load()
        cases = sorted(df["case_name"].dropna().unique().tolist())
        hidden_cases = set(st.session_state.hidden_cases)
        
        if not cases:
            st.info("No cases available.")
        else:
            b1, b2, b3, b4 = st.columns([2, 1, 1, 1])
            target_case = b1.selectbox("Select Case/Book", cases)
            
            if b2.button("🙈 Hide Case"):
                hidden_cases.add(target_case)
                st.session_state.hidden_cases = list(hidden_cases)
                save_session_state()
                st.success(f"Hidden: {target_case}")
                time.sleep(0.2)
                st.rerun()
            
            if b3.button("👁️ Show All"):
                st.session_state.hidden_cases = []
                save_session_state()
                st.success("All cases are now visible.")
                time.sleep(0.2)
                st.rerun()
            
            if b4.button("🗑️ Delete Case", type="secondary"):
                if st.button(f"⚠️ Confirm Delete '{target_case}'", key="confirm_delete_case"):
                    with st.spinner(f"Deleting '{target_case}'..."):
                        case_dir = DATA_ROOT / safe_book_name(target_case)
                        if case_dir.exists():
                            shutil.rmtree(case_dir, ignore_errors=True)
                        
                        clinical_case_dir = CLINICAL_DIR / safe_book_name(target_case.replace("Clinical::", ""))
                        if clinical_case_dir.exists():
                            shutil.rmtree(clinical_case_dir, ignore_errors=True)
                        
                        d = md_load()
                        case_images = d[d["case_name"] == target_case]["image_path"].tolist()
                        d = d[d["case_name"] != target_case]
                        md_save_immediate(d)
                        
                        st.session_state.edited_images_set -= set(case_images)
                        for img_path in case_images:
                            st.session_state.last_visit_times.pop(img_path, None)
                            st.session_state.edit_timestamps.pop(img_path, None)
                        
                    st.success(f"✅ Deleted case: {target_case}")
                    time.sleep(0.2)
                    st.rerun()

            if hidden_cases:
                st.markdown("**Hidden Cases:**")
                for case in sorted(hidden_cases):
                    col_case, col_unhide = st.columns([3, 1])
                    col_case.text(case)
                    if col_unhide.button("👁️", key=f"unhide_{case}"):
                        hidden_cases.remove(case)
                        st.session_state.hidden_cases = list(hidden_cases)
                        save_session_state()
                        st.rerun()

        st.markdown("---")
        st.markdown("### 🏷️ Tag Library Management")

        custom_tags = st.session_state.get("custom_tags", {})

        if custom_tags:
            st.metric("Total Custom Tags", len(custom_tags))
            
            if st.button("📤 Export Tag Library"):
                tags_json = json.dumps(custom_tags, indent=2)
                st.download_button(
                    label="Download Tags JSON",
                    data=tags_json,
                    file_name=f"custom_tags_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            uploaded_tags = st.file_uploader("📥 Import Tag Library", type=["json"])
            if uploaded_tags:
                try:
                    imported_tags = json.loads(uploaded_tags.read())
                    st.session_state.custom_tags.update(imported_tags)
                    save_tags()
                    st.success(f"✅ Imported {len(imported_tags)} tags!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
        else:
            st.info("No custom tags yet. Create your first tag by editing an image!")

        st.markdown("---")
        st.markdown("### 📦 Export System")
        
        exp1, exp2 = st.columns([2, 1])
        out_base = exp1.text_input("Export directory", value=str(DATA_ROOT / "_export"))
        
        if exp2.button("📤 EXPORT BY STRUCTURE"):
            with st.spinner("Exporting images..."):
                out_root = Path(out_base)
                copied = 0
                df_export = md_load()
                
                for _, r in df_export.iterrows():
                    site = (r.get("anatomical_site") or "unknown").strip() or "unknown"
                    lesion = (r.get("lesion_type") or "unknown").strip() or "unknown"
                    case = (r.get("case_name") or "unknown").strip() or "unknown"
                    
                    edit_info = get_image_edit_info(r["image_path"])
                    status = "edited" if edit_info["is_edited"] else "unedited"
                    
                    dst_dir = out_root / site / lesion / status / safe_book_name(case)
                    ensure_dir(dst_dir)
                    src = DATA_ROOT / str(r["image_path"])
                    
                    if src.exists():
                        try:
                            shutil.copy2(src, unique_filename(dst_dir / src.name))
                            copied += 1
                        except Exception:
                            continue
            
            st.success(f"✅ Exported {copied} images → {out_root}")
            st.info("Structure: Site/Lesion/EditStatus/Case/")

        with st.expander("📊 Advanced Export Options", expanded=False):
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("📋 Export Review Report"):
                    report = generate_analytics_report(df)
                    st.download_button(
                        label="Download Review Report",
                        data=report,
                        file_name=f"review_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                
                if st.button("🏷️ Export Tags Summary"):
                    tags_summary = generate_tags_summary(df)
                    st.download_button(
                        label="Download Tags Summary",
                        data=tags_summary,
                        file_name=f"tags_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with export_col2:
                if st.button("📈 Export Edit History"):
                    edit_history = generate_edit_history()
                    st.download_button(
                        label="Download Edit History",
                        data=edit_history,
                        file_name=f"edit_history_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
                if st.button("🔍 Export Unedited List"):
                    unedited_list = generate_unedited_list(df)
                    st.download_button(
                        label="Download Unedited Images List",
                        data=unedited_list,
                        file_name=f"unedited_images_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

        st.markdown("---")
        st.markdown("### 💾 Backup & Restore System")
        
        backup_col1, backup_col2 = st.columns(2)
        
        with backup_col1:
            st.markdown("**Create Backup**")
            if st.button("💾 Full Backup", type="primary"):
                backup_name = f"oral_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_dir = DATA_ROOT.parent / backup_name
                
                with st.spinner("Creating full backup..."):
                    try:
                        shutil.copytree(DATA_ROOT, backup_dir)
                        st.success(f"✅ Full backup created: {backup_dir}")
                    except Exception as e:
                        st.error(f"Backup failed: {e}")
            
            if st.button("📊 Metadata Only Backup"):
                backup_name = f"metadata_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                backup_path = DATA_ROOT.parent / backup_name
                
                with st.spinner("Creating metadata backup..."):
                    try:
                        import zipfile
                        with zipfile.ZipFile(backup_path, 'w') as zipf:
                            if PARQUET_PATH.exists():
                                zipf.write(PARQUET_PATH, PARQUET_PATH.name)
                            if CSV_PATH.exists():
                                zipf.write(CSV_PATH, CSV_PATH.name)
                            if SESSION_STATE_JSON.exists():
                                zipf.write(SESSION_STATE_JSON, SESSION_STATE_JSON.name)
                            if WORK_SESSION_JSON.exists():
                                zipf.write(WORK_SESSION_JSON, WORK_SESSION_JSON.name)
                        st.success(f"✅ Metadata backup created: {backup_path}")
                    except Exception as e:
                        st.error(f"Metadata backup failed: {e}")
        
        with backup_col2:
            st.markdown("**Restore from Backup**")
            backup_dirs = [d for d in DATA_ROOT.parent.iterdir() 
                         if d.is_dir() and d.name.startswith("oral_backup_")]
            
            if backup_dirs:
                selected_backup = st.selectbox("Select Backup", 
                                             [d.name for d in sorted(backup_dirs, reverse=True)])
                
                if st.button("🔄 Restore from Backup", type="secondary"):
                    if st.button("⚠️ Confirm Restore", key="confirm_restore"):
                        backup_path = DATA_ROOT.parent / selected_backup
                        with st.spinner("Restoring from backup..."):
                            try:
                                if DATA_ROOT.exists():
                                    shutil.rmtree(DATA_ROOT)
                                shutil.copytree(backup_path, DATA_ROOT)
                                
                                load_persistent_data()
                                md_load_cached.clear()
                                
                                st.success("✅ Successfully restored from backup!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Restore failed: {e}")
            else:
                st.info("No backups found.")

        st.markdown("---")
        st.markdown("### ℹ️ System Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("**Data Storage:**")
            st.code(f"Root: {DATA_ROOT}")
            st.code(f"Clinical: {CLINICAL_DIR}")
            
            total_size = sum(f.stat().st_size for f in DATA_ROOT.rglob('*') if f.is_file())
            st.metric("Storage Used", f"{total_size / (1024**2):.1f} MB")
        
        with info_col2:
            st.markdown("**File Formats:**")
            st.code(f"Metadata: {'Parquet' if _parquet_available() else 'CSV'}")
            
            if os.name == "nt":
                if st.button("📂 Open Data Directory"):
                    try:
                        os.startfile(str(DATA_ROOT))
                    except Exception:
                        st.warning("Could not open directory.")

# ========================= HELPER FUNCTIONS =========================
def generate_tags_summary(df: pd.DataFrame) -> str:
    """Generate tags usage summary CSV"""
    all_tags = []
    for tags_str in df['tags'].dropna():
        if tags_str:
            all_tags.extend([t.strip() for t in tags_str.split(',')])
    
    tag_counts = pd.Series(all_tags).value_counts()
    return tag_counts.to_csv()

def generate_edit_history() -> str:
    """Generate edit history JSON"""
    edit_data = {
        "session_id": st.session_state.work_session_manager.session_id,
        "total_edited": len(st.session_state.edited_images_set),
        "edit_timestamps": st.session_state.edit_timestamps,
        "last_visit_times": st.session_state.last_visit_times,
        "productivity_stats": st.session_state.productivity_stats,
        "exported_at": datetime.now().isoformat()
    }
    return json.dumps(edit_data, indent=2)

def generate_unedited_list(df: pd.DataFrame) -> str:
    """Generate CSV of unedited images"""
    edited_images = st.session_state.edited_images_set
    unedited = df[~df["image_path"].isin(edited_images)]
    
    export_cols = ['case_name', 'image_path', 'anatomical_site', 'lesion_type', 
                  'confidence', 'relevance_score', 'caption']
    return unedited[export_cols].to_csv(index=False)

if __name__ == "__main__":
    main()