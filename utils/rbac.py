"""rag_agent/utils/rbac.py — RBAC 权限过滤构建器
核心：根据用户身份生成 ChromaDB metadata_filter，实现同名问题不同视图。"""
from typing import Optional


# ── 可见性等级（从低到高）────────────────────
VISIBILITY_LADDER = ["public", "internal", "confidential", "secret"]


def get_clearance_rank(role: str, level: str) -> int:
    """返回当前用户的访问权限等级索引。"""
    # ponytail: role → clearance 映射表，生产环境可从 JWT / LDAP 读取
    ROLE_LEVELS = {
        "client": "public",
        "sales": "internal",
        "engineer": "internal",
        "manager": "confidential",
        "admin": "secret",
    }
    user_level = ROLE_LEVELS.get(role, "public")
    if VISIBILITY_LADDER.index(user_level) >= VISIBILITY_LADDER.index(level):
        return 1  # ✅ 允许访问
    return 0  # ❌ 禁止访问


def build_filter(
    user_role: str = "client",
    clearance_level: str = "public",
    department: Optional[str] = None,
    expire_after: Optional[str] = None,
) -> dict:
    """构建 ChromaDB metadata_filter。

    Args:
        user_role: 用户角色 (client/sales/engineer/manager/admin)
        clearance_level: 查询要求的最低密级 (public/internal/confidential/secret)
        department: 部门过滤（如 'hr'、'finance'），None 则不限
        expire_after: 只查此日期之后的文档 (ISO 格式 YYYY-MM-DD)

    Returns:
        dict 可直接传入 chromadb.Collection.query(filter=...)

    Example:
        >>> build_filter("engineer", "internal", "production")
        {'$and': [{'dept': 'production'}, {'visibility': {'$gte': 'internal'}}]}
    """
    conditions = []

    # 1. 权限过滤
    can_access = get_clearance_rank(user_role, clearance_level)
    if can_access == 0:
        # ponytail: 无权限时直接返回一个永远不匹配的空过滤器，防止数据泄露
        return {"_impossible": True}

    conditions.append({"visibility": {"$gte": clearance_level}})

    # 2. 部门隔离
    if department:
        conditions.append({"dept": department})

    # 3. 时间衰减：过期文档不参与检索
    if expire_after:
        conditions.append({"created_at": {"$gte": expire_after}})

    return {"$and": conditions} if conditions else {}


if __name__ == "__main__":
    print("=== RBAC Filter Tests ===")

    # Test 1: 客户只能看 public
    f = build_filter("client", "public")
    assert f != {"_impossible": True}, "客户应能看到 public"
    print(f"✅ client → public: {f}")

    # Test 2: 客户试图查 confidential → 空结果
    f = build_filter("client", "confidential")
    assert f == {"_impossible": True}, "客户无权查机密"
    print(f"✅ client → confidential: BLOCKED")

    # Test 3: 工程师看 internal + 指定部门
    f = build_filter("engineer", "internal", "production")
    assert "dept" in str(f), "应包含部门过滤"
    print(f"✅ engineer + dept: {f}")

    # Test 4: 时间过滤
    f = build_filter("admin", "secret", expire_after="2026-09-01")
    assert "created_at" in str(f), "应包含时间过滤"
    print(f"✅ with expire_after: {f}")

    print("\n✅ All RBAC tests passed.")
