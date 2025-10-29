import os
import json
import psycopg2
from psycopg2.extras import execute_values

DB_CONFIG = {
    "host": "aws-1-us-west-1.pooler.supabase.com",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres.rsrrttxgsnnnsjegfyxh",
    "password": "Proje12338!",
    "sslmode": "require",
}

SCHEMA_NAME = "public"
TABLE_NAME = "lgs_questions"
JSON_PATH = os.path.join(os.path.dirname(__file__), "2017-2023_LGS_İngilizce-Tüm-Sorular.json")

RUN_DDL = False
UPSERT_ENABLED = False

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    year INTEGER NOT NULL,
    question_number TEXT NOT NULL,
    question_text TEXT NOT NULL,
    option_a TEXT NOT NULL,
    option_b TEXT NOT NULL,
    option_c TEXT NOT NULL,
    option_d TEXT NOT NULL,
    correct_option TEXT NOT NULL,
    topic TEXT NOT NULL
);
"""

INSERT_SQL_UPSERT = f"""
INSERT INTO {SCHEMA_NAME}.{TABLE_NAME} (
    year, question_number, question_text,
    option_a, option_b, option_c, option_d,
    correct_option, topic
) VALUES %s
ON CONFLICT (id) DO NOTHING;
"""

INSERT_SQL_SIMPLE = f"""
INSERT INTO {SCHEMA_NAME}.{TABLE_NAME} (
    year, question_number, question_text,
    option_a, option_b, option_c, option_d,
    correct_option, topic
) VALUES %s;
"""

def load_rows_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        correct_opt = str(item["correct_option"]).strip().upper()[:1]
        qn_text = str(item["question_number"]).strip()
        rows.append(
            (
                int(item["year"]),
                qn_text,
                item["question_text"],
                item["option_a"],
                item["option_b"],
                item["option_c"],
                item["option_d"],
                correct_opt,
                item["topic"],
            )
        )
    return rows


def main():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON bulunamadı: {JSON_PATH}")

    rows = load_rows_from_json(JSON_PATH)

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            if RUN_DDL:
                cur.execute(CREATE_TABLE_SQL)
            insert_sql = INSERT_SQL_UPSERT if UPSERT_ENABLED else INSERT_SQL_SIMPLE
            if rows:
                execute_values(cur, insert_sql, rows, page_size=500)
        conn.commit()
        print(f"{len(rows)} satır işlendi ve {SCHEMA_NAME}.{TABLE_NAME} tablosuna eklendi.")
    except Exception as exc:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
