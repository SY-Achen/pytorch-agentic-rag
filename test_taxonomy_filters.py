import server


def test_upload_metadata_normalizes_public_source_fields():
    meta = server._document_taxonomy(
        course="数据结构", chapter="树", topic="二叉树遍历",
        source_url="https://example.edu/ds", license_name="CC BY-NC-SA 4.0", version="2026.1",
    )
    assert meta["topic"] == "二叉树遍历"
    assert meta["source_url"] == "https://example.edu/ds"


def test_course_filter_combines_with_existing_acl_filter():
    result = server._merge_where_filters({"access": "public"}, "数据结构", "树")
    assert result == {"$and": [{"access": "public"}, {"course": "数据结构"}, {"chapter": "树"}]}
