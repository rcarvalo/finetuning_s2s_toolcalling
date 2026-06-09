import pytest

from s2s_toolcalling.tools.database import UnsafeQueryError, ensure_read_only


def test_select_allowed():
    assert ensure_read_only("SELECT * FROM employees").startswith("SELECT")


def test_with_cte_allowed():
    sql = "WITH today AS (SELECT * FROM appointments) SELECT * FROM today"
    assert ensure_read_only(sql)


def test_trailing_semicolon_stripped():
    assert ensure_read_only("SELECT 1;") == "SELECT 1"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM employees",
        "DROP TABLE appointments",
        "INSERT INTO employees VALUES (1)",
        "UPDATE employees SET full_name = 'x'",
        "SELECT 1; DROP TABLE employees",
        "SELECT pg_sleep(10); DELETE FROM employees",
        "TRUNCATE employees",
        "CREATE TABLE evil (id int)",
        "GRANT ALL ON employees TO public",
        "SET ROLE postgres",
        "",
        "   ",
    ],
)
def test_rejected(sql):
    with pytest.raises(UnsafeQueryError):
        ensure_read_only(sql)


def test_comment_hidden_write_rejected():
    with pytest.raises(UnsafeQueryError):
        ensure_read_only("SELECT 1 /* */ ; DELETE FROM employees")


def test_forbidden_keyword_inside_select_rejected():
    # conservateur : un SELECT contenant un mot-clé d'écriture est refusé
    with pytest.raises(UnsafeQueryError):
        ensure_read_only("SELECT * FROM employees WHERE note = 'do not delete'")
