from app.retrieval.hybrid import Document, HybridRetriever


def test_hybrid_retrieval_ranks_preprocessing_function_first():
    retriever = HybridRetriever(
        [
            Document("1", "normalize.js", "function normalize(str) { const str2 = str.trim(); return forward(str2); }", "normalize"),
            Document("2", "locale.js", "function check(s) { var pre = s.slice(0, 6); return pre === 'en-US'; }", "check"),
            Document("3", "perf.js", "function perf(str) { if (act(A, str)) { return act(B, str); } }", "perf"),
        ]
    )
    results = retriever.search("How is the input preprocessed before going to the main function?")
    assert [result.document.document_id for result in results] == ["1", "2", "3"]


def test_exact_symbol_match_beats_lexical_noise():
    retriever = HybridRetriever(
        [
            Document("caller", "a.js", "return getUserData(user);", "caller"),
            Document("definition", "b.js", "function getUserData(user) { return user.data; }", "getUserData"),
        ]
    )
    assert retriever.search("get_user_data")[0].document.document_id == "definition"
