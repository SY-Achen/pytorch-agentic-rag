"""Upload access control: public/private/shared + retrieval filtering."""
import server


def test_parse_access_defaults():
    a, u = server._parse_access(None, None)
    assert a == "public"
    assert u == ""


def test_parse_access_private():
    a, u = server._parse_access("private", None)
    assert a == "private"
    assert u == ""


def test_parse_access_shared_normalizes_users():
    a, u = server._parse_access("shared", " alice , bob ,alice ")
    assert a == "shared"
    assert u == "alice,bob"


def test_parse_access_shared_without_users_falls_back_public():
    a, u = server._parse_access("shared", "  ")
    assert a == "public"
    assert u == ""


def test_acl_where_public_visible_to_anyone():
    w = server._acl_where("someone")
    assert {"access": "public"} in w["$or"]


def test_acl_where_owner_visible_to_owner():
    w = server._acl_where("mine_user")
    assert {"owner": "mine_user"} in w["$or"]


def test_acl_where_allowed_users_visible():
    w = server._acl_where("alice")
    assert {"allowed_users": {"$contains": "alice"}} in w["$or"]