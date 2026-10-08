"""User database tests: register -> login via SQLite, replacing MOCK_USERS lookups."""
import server


def _reset_db(tmp_path):
    server.SESSION_DB = tmp_path / "test.db"
    server._init_db()


def test_register_then_login_ok(tmp_path):
    _reset_db(tmp_path)
    req = server.LoginRequest(username="student001", password="exam123")
    reg = server.register(server.RegisterRequest(
        username="student001", password="exam123", name="王同学", dept="education"))
    assert reg.get("success") is True
    # New registrations require administrator approval before login.
    resp = server.login(req)
    assert resp.status_code == 403


def test_seeded_users_have_governance_metadata(tmp_path):
    _reset_db(tmp_path)
    for username, role in (("wangwu", "admin"), ("zhangsan", "student"), ("lisi", "student")):
        user = server._user_from_db(username)
        assert user["role"] == role
        assert user["status"] == "approved"


def test_duplicate_username_rejected(tmp_path):
    _reset_db(tmp_path)
    r1 = server.register(server.RegisterRequest(
        username="dup001", password="pw123456", name="A", dept="education"))
    assert r1.get("success") is True
    r2 = server.register(server.RegisterRequest(
        username="dup001", password="other678", name="B", dept="education"))
    assert hasattr(r2, "status_code") and r2.status_code == 409


def test_wrong_password_rejected(tmp_path):
    _reset_db(tmp_path)
    server.register(server.RegisterRequest(
        username="stu002", password="right123", name="X", dept="education"))
    resp = server.login(server.LoginRequest(username="stu002", password="wrong!"))
    assert hasattr(resp, "status_code") and resp.status_code == 401