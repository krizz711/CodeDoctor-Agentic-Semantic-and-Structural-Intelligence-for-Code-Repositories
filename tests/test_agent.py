from __future__ import annotations

from app.agent.code_agent import AgentResponse, CodeDoctorAgent


def test_agent_without_llm():
    agent = CodeDoctorAgent()
    evidence = [
        {
            "file_path": "auth.js",
            "start_line": 1,
            "end_line": 5,
            "type": "FUNCTION",
            "name": "loginUser",
            "content": "function loginUser(email) { return db.find(email); }",
        }
    ]
    resp = agent.ask("What does loginUser do?", evidence)
    assert isinstance(resp, AgentResponse)
    assert resp.question == "What does loginUser do?"
    assert len(resp.evidence) == 1
    assert resp.grounded is True


def test_agent_explain_symbol():
    agent = CodeDoctorAgent()
    definition = {
        "file_path": "auth.js",
        "start_line": 1,
        "end_line": 5,
        "type": "FUNCTION",
        "name": "login",
        "content": "function login() {}",
    }
    callers = [
        {
            "file_path": "app.js",
            "start_line": 10,
            "end_line": 15,
            "type": "FUNCTION",
            "name": "main",
            "content": "login()",
        }
    ]
    resp = agent.explain_symbol("login", definition, callers, [])
    assert "login" in resp.question


def test_agent_review_code():
    agent = CodeDoctorAgent()
    chunks = [
        {
            "file_path": "auth.js",
            "start_line": 1,
            "end_line": 3,
            "type": "FUNCTION",
            "name": "auth",
            "content": "function auth(pwd) { eval(pwd); }",
        }
    ]
    resp = agent.review_code(chunks, focus="security")
    assert "security" in resp.question
