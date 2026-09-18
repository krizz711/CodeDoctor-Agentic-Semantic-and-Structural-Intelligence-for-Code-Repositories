from app.indexing.sqlite_index import SQLiteIndex, normalize_identifier


def test_normalize_identifier_variants_match():
    assert normalize_identifier("getUserData") == "getuserdata"
    assert normalize_identifier("get_user_data") == "getuserdata"
    assert normalize_identifier("GET_USER_DATA") == "getuserdata"


def test_symbol_index_returns_exact_named_symbols():
    index = SQLiteIndex(path=':memory:')
    index.index_file(
        'src/auth.js',
        """
        export function loginUser(email, password) {
            return { email, password };
        }

        class AuthService {
            async getUserData() {
                return { ok: true };
            }
        }
        """,
        commit_hash='abc123',
        language='javascript',
    )

    matches = index.search_symbols('loginUser')
    assert matches
    assert any(item['name'] == 'loginUser' for item in matches)

    class_matches = index.search_symbols('get_user_data')
    assert class_matches
    assert any(item['name'] == 'getUserData' for item in class_matches)
