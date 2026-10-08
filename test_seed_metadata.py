import server


def test_seed_metadata_contains_provenance_fields():
    meta = server._seed_metadata(
        source="DataStructure__5-tree.md", course="数据结构", chapter="树",
        source_url="https://example.edu/tree", license_name="CC BY 4.0", version="2026.1",
    )
    assert meta["type"] == "seed"
    assert meta["topic"] == "树"
    assert meta["source_url"] == "https://example.edu/tree"
    assert meta["license"] == "CC BY 4.0"
