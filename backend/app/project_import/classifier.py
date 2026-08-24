from __future__ import annotations

from pathlib import PurePath

from ..db.models import AssetType, ProjectFileRole

_ROLE_KEYWORDS: dict[ProjectFileRole, tuple[str, ...]] = {
    ProjectFileRole.TENDER: ("招标", "采购文件", "询价文件", "招标资料", "tender", "rfp"),
    ProjectFileRole.BID: ("投标", "响应文件", "应答文件", "报价文件", "商务标", "技术标", "bid", "proposal"),
    ProjectFileRole.ATTACHMENT: ("附件", "附录", "补充材料", "attachment", "appendix"),
}

_DRAWING_KEYWORDS = (
    "图纸",
    "招标图",
    "技术图",
    "施工图",
    "总图",
    "示意图",
    "布置图",
    "流程图",
    "drawing",
    "blueprint",
)


def _matches(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(keyword.casefold() in normalized for keyword in keywords)


def classify_project_file_role(filename: str, parent_parts: tuple[str, ...] = ()) -> ProjectFileRole:
    """Classify conservatively, giving directory context more weight than filenames."""
    parent_text = " ".join(parent_parts)
    scores = {
        role: (3 if _matches(parent_text, keywords) else 0) + (2 if _matches(filename, keywords) else 0)
        for role, keywords in _ROLE_KEYWORDS.items()
    }
    best_score = max(scores.values(), default=0)
    winners = [role for role, score in scores.items() if score == best_score and score > 0]
    return winners[0] if len(winners) == 1 else ProjectFileRole.UNKNOWN


def classify_asset_type(filename: str, parent_parts: tuple[str, ...] = ()) -> AssetType:
    """Classify supported files without guessing whether a document is scanned."""
    path_text = " ".join((*parent_parts, PurePath(filename).stem))
    if _matches(path_text, _DRAWING_KEYWORDS):
        return AssetType.TECHNICAL_DRAWING
    return AssetType.DOCUMENT
