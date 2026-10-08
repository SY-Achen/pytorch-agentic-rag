from fastapi.testclient import TestClient
import server


def _client_with_auth(tmp_path):
    server.SESSION_DB = tmp_path / 'private-data.db'
    server._init_db()
    client = TestClient(server.app)
    tokens = {}
    for user in ('zhangsan', 'lisi', 'wangwu'):
        tokens[user] = client.post('/api/login', json={'username':user,'password':'123'}).json()['token']
    return client, tokens


def test_history_requires_auth_and_cannot_select_another_user(tmp_path):
    client, tokens = _client_with_auth(tmp_path)
    assert client.get('/api/session/history').status_code == 401
    server._log_session('lisi','engineering','secret question','secret answer','[]',trace_id='tr_lisi_1')
    response = client.get('/api/session/history?username=lisi', headers={'Authorization':f"Bearer {tokens['zhangsan']}"})
    assert response.status_code == 200
    assert response.json()['history'] == []


def test_trace_requires_owner_or_admin(tmp_path):
    client, tokens = _client_with_auth(tmp_path)
    assert client.get('/api/trace/tr_zhangsan_1').status_code == 401
    student={'Authorization':f"Bearer {tokens['zhangsan']}"}
    assert client.get('/api/trace/tr_lisi_1',headers=student).status_code == 403
    assert client.get('/api/trace/tr_zhangsan_1',headers=student).status_code == 200


def test_feedback_uses_authenticated_identity_and_session_owner(tmp_path):
    client, tokens = _client_with_auth(tmp_path)
    assert client.post('/api/feedback',json={'username':'zhangsan','session_id':1,'thumbs_up':True}).status_code == 401
    server._log_session('lisi','engineering','q','a','[]')
    session_id=server.sqlite3.connect(str(server.SESSION_DB)).execute('select id from sessions').fetchone()[0]
    response=client.post('/api/feedback',headers={'Authorization':f"Bearer {tokens['zhangsan']}"},json={'username':'lisi','session_id':session_id,'thumbs_up':True})
    assert response.status_code == 403
