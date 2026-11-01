def test_register_and_me(client):
    r = client.post("/auth/register", json={"name": "Asha", "email": "Asha@Example.com", "password": "supersecret1"})
    assert r.status_code == 201
    body = r.json()
    assert body["user"]["role"] == "farmer" and body["user"]["email"] == "asha@example.com"
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200 and me.json()["name"] == "Asha"


def test_duplicate_email_rejected(client):
    p = {"name": "Dup", "email": "dup@example.com", "password": "password123"}
    assert client.post("/auth/register", json=p).status_code == 201
    assert client.post("/auth/register", json=p).status_code == 409


def test_weak_password_and_bad_email_rejected(client):
    assert client.post("/auth/register", json={"name": "A B", "email": "a@b.com", "password": "short"}).status_code == 422
    assert client.post("/auth/register", json={"name": "A B", "email": "not-an-email", "password": "password123"}).status_code == 422


def test_login_success_and_failure(client):
    client.post("/auth/register", json={"name": "Login T", "email": "login@example.com", "password": "password123"})
    ok = client.post("/auth/login", json={"email": "login@example.com", "password": "password123"})
    assert ok.status_code == 200 and ok.json()["token_type"] == "bearer"
    assert client.post("/auth/login", json={"email": "login@example.com", "password": "wrongpass1"}).status_code == 401
    assert client.post("/auth/login", json={"email": "nobody@example.com", "password": "password123"}).status_code == 401


def test_protected_routes_need_token(client, payload):
    assert client.post("/predict", json=payload).status_code == 401
    assert client.get("/history").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_role_based_access(client, farmer, admin):
    assert client.get("/admin/stats", headers=farmer).status_code == 403
    assert client.get("/admin/stats", headers=admin).status_code == 200
    assert client.get("/admin/model-metrics", headers=admin).status_code == 200
