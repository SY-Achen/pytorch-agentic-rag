"""Admin-only upload and document management behavior."""
import server


def test_approved_admin_is_allowed_to_upload():
    assert server._is_admin({"status": "approved", "role": "admin"})


def test_approved_student_is_not_allowed_to_upload():
    assert not server._is_admin({"status": "approved", "role": "student"})


def test_pending_admin_is_not_allowed_to_upload():
    assert not server._is_admin({"status": "pending", "role": "admin"})
