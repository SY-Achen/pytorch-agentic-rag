"""Governance regression tests: account approval, metadata and CORS parsing."""
import server
from fastapi.testclient import TestClient


def test_cors_origins_parse_whitelist_and_drop_wildcard():
    assert server._parse_cors_origins("http://localhost:3000, https://demo.example.com, *") == [
        "http://localhost:3000", "https://demo.example.com"
    ]


def test_document_taxonomy_keeps_course_chapter_topic_and_license():
    meta = server._document_taxonomy(
        course="数据结构", chapter="树", topic="二叉树遍历",
        source_url="https://example.edu/ds", license_name="CC BY-NC-SA 4.0", version="2026.1",
    )
    assert meta == {
        "course": "数据结构", "chapter": "树", "topic": "二叉树遍历",
        "source_url": "https://example.edu/ds", "license": "CC BY-NC-SA 4.0", "version": "2026.1",
    }


def test_user_state_requires_approved_status():
    assert server._can_use_service({"status": "approved", "role": "student"}) is True
    assert server._can_use_service({"status": "pending", "role": "student"}) is False
    assert server._is_admin({"status": "approved", "role": "admin"}) is True
    assert server._is_admin({"status": "approved", "role": "student"}) is False


def test_admin_can_list_and_approve_pending_user(tmp_path):
    server.SESSION_DB = tmp_path / "test.db"
    server._init_db()
    client = TestClient(server.app)
    assert client.post("/api/register", json={
        "username": "pending001", "password": "exam123", "name": "待审", "dept": "education"
    }).json()["success"] is True
    admin_token = client.post("/api/login", json={"username": "wangwu", "password": "123"}).json()["token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    pending = client.get("/api/admin/users/pending", headers=headers)
    assert pending.status_code == 200
    assert any(u["username"] == "pending001" for u in pending.json()["users"])
    approved = client.post("/api/admin/users/pending001/approve", headers=headers)
    assert approved.status_code == 200
    assert client.post("/api/login", json={"username": "pending001", "password": "exam123"}).status_code == 200


def test_student_cannot_upload_knowledge(tmp_path):
    server.SESSION_DB = tmp_path / "test.db"
    server._init_db()
    client = TestClient(server.app)
    token = client.post("/api/login", json={"username": "zhangsan", "password": "123"}).json()["token"]
    response = client.post(
        "/api/knowledge/upload", headers={"Authorization": f"Bearer {token}"},
        files={"file": ("notes.txt", b"content", "text/plain")},
    )
    assert response.status_code == 403
