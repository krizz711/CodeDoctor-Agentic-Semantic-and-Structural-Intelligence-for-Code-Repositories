from app.indexing.sqlite_index import SQLiteIndex


def test_sqlite_fts5_indexing_and_search():
    index = SQLiteIndex(":memory:")
    code1 = """
    export function processPayment(amount, currency) {
        validateCurrency(currency);
        return gateway.charge(amount);
    }
    """
    code2 = """
    export function renderHeader(title) {
        return `<h1>${title}</h1>`;
    }
    """
    index.index_file("src/payment.js", code1)
    index.index_file("src/ui.js", code2)

    # Search for payment
    results = index.search_fts("processPayment gateway")
    assert results
    assert results[0]["file_path"] == "src/payment.js"

    # Search for header
    results_ui = index.search_fts("renderHeader")
    assert results_ui
    assert results_ui[0]["file_path"] == "src/ui.js"


def test_sqlite_reindex_does_not_leak_fts_rows():
    index = SQLiteIndex(":memory:")
    initial_code = "function originalFunc() { return 1; }"
    index.index_file("src/test.js", initial_code)

    assert len(index.search_fts("originalFunc")) > 0

    # Reindex with changed content
    updated_code = "function updatedFunc() { return 2; }"
    index.index_file("src/test.js", updated_code)

    # Old symbol should NOT be in FTS anymore
    assert len(index.search_fts("originalFunc")) == 0
    # New symbol SHOULD be in FTS
    assert len(index.search_fts("updatedFunc")) > 0
