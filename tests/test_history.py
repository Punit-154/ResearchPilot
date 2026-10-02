import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_history_imports_cleanly():
    """Verify that adding history routes doesn't break the app import"""
    response = client.get("/health")
    assert response.status_code == 200

# Testing history logic with mocked DB
class MockQueryHistory:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.id = 1
        self.created_at = "2026-10-02T12:00:00Z"

class MockQuery:
    def __init__(self, items):
        self.items = items
        
    def order_by(self, *args):
        return self
        
    def filter(self, *args):
        return self
        
    def all(self):
        return self.items

    def first(self):
        return self.items[0] if self.items else None

class MockSession:
    def __init__(self):
        self.added = []
    def query(self, model):
        if model.__name__ == 'QueryHistory':
            history = MockQueryHistory(
                question="Test question",
                paper_ids=[1],
                paper_titles=["Test Paper"],
                full_response={"answer": "Test answer"},
                is_baseline=False
            )
            return MockQuery([history])
        return MockQuery([])
    def add(self, item):
        self.added.append(item)
    def commit(self):
        pass

def test_history_mocked_logic():
    # Test the logic that would be inside the routes using our mock session
    from backend.api.routes.history import get_history, get_history_detail
    
    db = MockSession()
    
    # Test list
    res = get_history(paper_id=None, db=db)
    assert "history" in res
    assert len(res["history"]) == 1
    assert res["history"][0]["question"] == "Test question"
    assert res["history"][0]["paper_ids"] == [1]
    
    # Test detail
    res_detail = get_history_detail(history_id=1, db=db)
    assert res_detail == {"answer": "Test answer"}
