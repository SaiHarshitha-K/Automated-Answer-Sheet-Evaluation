# db_utils.py
import pandas as pd

import os
import hashlib
import mysql.connector
from mysql.connector import Error


# ------------------------------------------------------------------
# Connection
# ------------------------------------------------------------------
def get_connection():
    """Return a new DB connection."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "omr_user"),
        password=os.getenv("DB_PASSWORD", "omr_user@123"),
        database=os.getenv("DB_NAME", "omr_portal"),
        autocommit=True,
    )




# db_utils.py (ADD)

def get_subject_name(subject_id: int):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM subjects WHERE id=%s LIMIT 1", (subject_id,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_students_for_exam_with_emails(*, exam_id: int, subject_id: int):
    """
    Returns:
    prn, exam_name, student_name, email, score, total_questions, lab_marks
    Email comes ONLY from students.email.
    Lab marks read from lab_marks table (if exists).
    """
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT
                es.prn,
                es.name AS exam_name,
                s.name AS student_name,
                s.email,
                es.score,
                es.total_questions
            FROM exam_students es
            LEFT JOIN students s
              ON TRIM(es.prn) = TRIM(s.prn)
            WHERE es.exam_id = %s
            """,
            (exam_id,),
        )
        rows = cur.fetchall() or []

        # add lab marks for each student (if any)
        for r in rows:
            prn = (r.get("prn") or "").strip()

            cur2 = conn.cursor(dictionary=True)
            cur2.execute(
                """
                SELECT marks
                FROM lab_marks
                WHERE subject_id=%s AND prn=%s
                LIMIT 1
                """,
                (subject_id, prn),
            )
            lm = cur2.fetchone()
            r["lab_marks"] = float(lm["marks"]) if lm and lm["marks"] is not None else None

        return rows
    finally:
        conn.close()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _hash_password(raw_password: str) -> str:
    return hashlib.sha256(raw_password.encode("utf-8")).hexdigest()

def _safe_execute(cur, stmt: str):
    """
    Runs SQL safely. If index already exists, ignore.
    Prevents crashing when init_db() runs again.
    """
    try:
        cur.execute(stmt)
    except Error as e:
        # ignore "Duplicate key name" / already exists
        msg = str(e).lower()
        if "duplicate key name" in msg or "already exists" in msg:
            return
        raise


def init_db():
    """Create tables if they do not exist."""
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS admins (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(64) UNIQUE NOT NULL,
            password_hash CHAR(64) NOT NULL,
            is_superadmin TINYINT(1) NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS batches (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(128) NOT NULL UNIQUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS courses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            batch_id INT NOT NULL,
            name VARCHAR(128) NOT NULL,
            UNIQUE (batch_id, name),
            FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS subjects (
            id INT AUTO_INCREMENT PRIMARY KEY,
            course_id INT NOT NULL,
            name VARCHAR(128) NOT NULL,
            UNIQUE (course_id, name),
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lab_marks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_id INT NOT NULL,
            prn VARCHAR(64) NOT NULL,
            marks DECIMAL(6,2) NULL,
            updated_by INT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_subject_prn (subject_id, prn),
            INDEX idx_lab_subject (subject_id),
            INDEX idx_lab_prn (prn),
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS students (
                id INT AUTO_INCREMENT PRIMARY KEY,
        prn VARCHAR(64) NOT NULL UNIQUE,
        name VARCHAR(128) NOT NULL,

        phone VARCHAR(20),
        email VARCHAR(128),

        password_hash CHAR(64) NOT NULL,

        batch_id INT NULL,
        course_id INT NULL,

        prn_last3 CHAR(3)
            GENERATED ALWAYS AS (RIGHT(prn, 3)) STORED,

        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

        CONSTRAINT fk_students_batch
            FOREIGN KEY (batch_id) REFERENCES batches(id)
            ON DELETE SET NULL,

        CONSTRAINT fk_students_course
            FOREIGN KEY (course_id) REFERENCES courses(id)
            ON DELETE SET NULL
            )
            """,
                """
        CREATE TABLE IF NOT EXISTS exams (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_id INT NOT NULL,
            pdf_path VARCHAR(255),
            key_path VARCHAR(255),
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS subject_publish (
            subject_id INT PRIMARY KEY,
            exam_id INT NOT NULL,
            is_published TINYINT(1) NOT NULL DEFAULT 0,
            published_by INT NULL,
            published_at DATETIME NULL,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
            FOREIGN KEY (exam_id) REFERENCES exams(id) ON DELETE CASCADE
        )
        """
        ,
        """
        CREATE TABLE IF NOT EXISTS exam_students (
            id INT AUTO_INCREMENT PRIMARY KEY,
            exam_id INT NOT NULL,
            prn VARCHAR(64) NOT NULL,
            name VARCHAR(128) NOT NULL,
            score INT NOT NULL,
            total_questions INT NOT NULL,
            image_path VARCHAR(255),
            is_conflict TINYINT DEFAULT 0,
            FOREIGN KEY (exam_id) REFERENCES exams(id) ON DELETE CASCADE
        )

        """,
        """
        CREATE TABLE IF NOT EXISTS exam_answers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            exam_student_id INT NOT NULL,
            question_no INT NOT NULL,
            student_answer VARCHAR(8),
            key_answer VARCHAR(8),
            is_correct TINYINT(1) NOT NULL,
            is_blank TINYINT(1) NOT NULL,
            UNIQUE (exam_student_id, question_no),
            FOREIGN KEY (exam_student_id) REFERENCES exam_students(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS question_papers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_id INT NOT NULL,
            qp_pdf_path VARCHAR(255),
            answer_key_path VARCHAR(255),
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS mcq_bank (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question_paper_id INT NOT NULL,
            question_no INT NOT NULL,
            question_text TEXT,

            option_a TEXT,
            option_b TEXT,
            option_c TEXT,
            option_d TEXT,

            correct_option CHAR(1),

            why_correct TEXT,
            why_a_wrong TEXT,
            why_b_wrong TEXT,
            why_c_wrong TEXT,
            why_d_wrong TEXT,

            UNIQUE (question_paper_id, question_no),
            FOREIGN KEY (question_paper_id) REFERENCES question_papers(id) ON DELETE CASCADE
        )
        """,

        # later you can extend for uploads/ocr results
    ]

    conn = get_connection()
    try:
        cur = conn.cursor()
        for stmt in ddl_statements:
            cur.execute(stmt)
        _safe_execute(cur, "CREATE INDEX idx_students_course_last3 ON students (course_id, prn_last3)")
        _safe_execute(cur, "CREATE INDEX idx_students_batch_course_last3 ON students (batch_id, course_id, prn_last3)")
        _safe_execute(cur, "CREATE INDEX idx_qp_subject_created ON question_papers (subject_id, created_at)")
        _safe_execute(cur, "CREATE INDEX idx_mcq_qp_qno ON mcq_bank (question_paper_id, question_no)")

    finally:
        conn.close()


# ------------------------------------------------------------------
# Admin operations
# ------------------------------------------------------------------
def any_admin_exists() -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM admins")
        (count,) = cur.fetchone()
        return count > 0
    finally:
        conn.close()


def create_admin(username: str, password: str, superadmin: bool = False) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO admins (username, password_hash, is_superadmin)
            VALUES (%s, %s, %s)
            """,
            (username, _hash_password(password), int(superadmin)),
        )
        return True
    except Error:
        # duplicate username etc.
        return False
    finally:
        conn.close()


def validate_admin(username: str, password: str):
    """Return (id, is_superadmin) if credentials are valid, else None."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, password_hash, is_superadmin FROM admins WHERE username=%s",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return None
        admin_id, stored_hash, is_superadmin = row
        if stored_hash == _hash_password(password):
            return admin_id, bool(is_superadmin)
        return None
    finally:
        conn.close()


def list_admins():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, username, is_superadmin FROM admins ORDER BY id")
        return cur.fetchall()
    finally:
        conn.close()


# ------------------------------------------------------------------
# Batch / Course / Subject operations
# ------------------------------------------------------------------
def add_batch(name: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO batches (name) VALUES (%s)", (name,))
        return True
    except Error:
        return False
    finally:
        conn.close()


def get_batches():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM batches ORDER BY id DESC")
        return cur.fetchall()
    finally:
        conn.close()


def add_course(batch_id: int, name: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO courses (batch_id, name) VALUES (%s, %s)",
            (batch_id, name),
        )
        return True
    except Error:
        return False
    finally:
        conn.close()


def get_courses(batch_id: int):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name FROM courses WHERE batch_id=%s ORDER BY id DESC",
            (batch_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def add_subject(course_id: int, name: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO subjects (course_id, name) VALUES (%s, %s)",
            (course_id, name),
        )
        return True
    except Error:
        return False
    finally:
        conn.close()


def get_subjects(course_id: int):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name FROM subjects WHERE course_id=%s ORDER BY id DESC",
            (course_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()

def _digits_only(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())

def _normalize_prn(prn: str) -> str:
    # keep ONLY digits, remove spaces/junk
    return _digits_only(str(prn)).strip()

# ------------------------------------------------------------------
# Student operations (NEW)
# ------------------------------------------------------------------

def upsert_student(
    *,
    prn: str,
    name: str,
    batch_id: int | None,
    course_id: int | None,
    phone: str | None = None,
    email: str | None = None,
    force_reset_password_to_prn: bool = False,
) -> bool:
    """
    Insert student if not exists, else update fields.
    Default password is PRN for new students.
    If force_reset_password_to_prn=True, reset password to PRN even if exists.
    """
    #prn = str(prn).strip()
    prn = _normalize_prn(prn)   # ✅ unified behavior
    name = str(name).strip()
    if not prn or not name:
        return False

    conn = get_connection()
    try:
        cur = conn.cursor()

        # check exists
        cur.execute("SELECT id FROM students WHERE prn=%s", (prn,))
        row = cur.fetchone()

        if row is None:
            # new student
            cur.execute(
                """
                INSERT INTO students (prn, name, phone, email, password_hash, batch_id, course_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    prn,
                    name,
                    phone,
                    email,
                    _hash_password(prn),   # ✅ default password = prn
                    batch_id,
                    course_id,
                ),
            )
        else:
            # update student
            if force_reset_password_to_prn:
                cur.execute(
                    """
                    UPDATE students
                    SET name=%s, phone=%s, email=%s, batch_id=%s, course_id=%s, password_hash=%s
                    WHERE prn=%s
                    """,
                    (
                        name,
                        phone,
                        email,
                        batch_id,
                        course_id,
                        _hash_password(prn),
                        prn,
                    ),
                )
            else:
                cur.execute(
                    """
                    UPDATE students
                    SET name=%s, phone=%s, email=%s, batch_id=%s, course_id=%s
                    WHERE prn=%s
                    """,
                    (
                        name,
                        phone,
                        email,
                        batch_id,
                        course_id,
                        prn,
                    ),
                )

        return True
    except Error:
        return False
    finally:
        conn.close()


def bulk_upsert_students_from_df(
    df,
    *,
    batch_id: int | None,
    course_id: int | None,
    prn_col: str = "PRN",
    name_col: str = "Name",
    phone_col: str | None = "Phone",
    email_col: str | None = "Email",
) -> dict:
    """
    df: pandas DataFrame
    Required columns: PRN, Name
    Optional: Phone, Email
    Returns summary dict.
    """
    inserted_or_updated = 0
    skipped = 0
    errors = []

    df.columns = [str(c).strip() for c in df.columns]

    # auto-detect PRN/Name if columns differ
    cols_lower = {c.lower(): c for c in df.columns}

    if prn_col not in df.columns:
        for cand in ["prn", "prn number", "prn_no", "student prn", "student_prn"]:
            if cand in cols_lower:
                prn_col = cols_lower[cand]
                break

    if name_col not in df.columns:
        for cand in ["name", "student name", "student_name"]:
            if cand in cols_lower:
                name_col = cols_lower[cand]
                break

    # optional safety check
    if prn_col not in df.columns or name_col not in df.columns:
        return {
            "inserted_or_updated": 0,
            "skipped": len(df),
            "errors": [f"Required columns not found. Found columns: {list(df.columns)}"],
        }
    for i, row in df.iterrows():
        try:
            raw_prn = row.get(prn_col, "")
            prn = _normalize_prn(raw_prn)   # ✅ digits only

            name = str(row.get(name_col, "")).strip()

            if not prn or not name.strip():
                skipped += 1
                continue

            phone = None
            email = None
            if phone_col and phone_col in df.columns:
                v = row.get(phone_col, None)
                phone = None if v is None or str(v).lower() == "nan" else str(v).strip()
            if email_col and email_col in df.columns:
                v = row.get(email_col, None)
                email = None if v is None or str(v).lower() == "nan" else str(v).strip()

            ok = upsert_student(
                prn=prn,
                name=name,
                phone=phone,
                email=email,
                batch_id=batch_id,
                course_id=course_id,
            )
            if ok:
                inserted_or_updated += 1
            else:
                skipped += 1
        except Exception as e:
            errors.append(f"Row {i}: {e}")

    return {
        "inserted_or_updated": inserted_or_updated,
        "skipped": skipped,
        "errors": errors,
    }


def validate_student(prn: str, password: str):
    """
    Return student row dict if login is valid else None.
    """
    prn = str(prn).strip()
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM students WHERE prn=%s",
            (prn,),
        )
        row = cur.fetchone()
        if not row:
            return None
        if row["password_hash"] == _hash_password(password):
            return row
        return None
    finally:
        conn.close()

def get_student_profile(prn: str):
    """
    Returns student profile + batch/course names.
    """
    prn = str(prn).strip()
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT
                s.prn, s.name, s.phone, s.email,
                s.batch_id, b.name AS batch_name,
                s.course_id, c.name AS course_name
            FROM students s
            LEFT JOIN batches b ON b.id = s.batch_id
            LEFT JOIN courses c ON c.id = s.course_id
            WHERE s.prn = %s
            LIMIT 1
            """,
            (prn,),
        )
        return cur.fetchone()
    finally:
        conn.close()

def get_student_by_prn(prn: str):
    prn = str(prn).strip()
    if not prn:
        return None
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT prn, name, batch_id, course_id FROM students WHERE prn=%s LIMIT 1",
            (prn,),
        )
        return cur.fetchone()
    finally:
        conn.close()

def get_course_id_for_subject(subject_id: int) -> int | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT course_id FROM subjects WHERE id=%s", (subject_id,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()
def get_course_batch_id(course_id: int) -> int | None:
    """
    Returns the batch_id for a given course.
    Used for scoping last3 PRN matching (batch+course safety).
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT batch_id FROM courses WHERE id=%s", (course_id,))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()

def get_latest_exam_id_for_course(course_id: int) -> int | None:
    """
    Returns latest exam_id for the given course (across all subjects in that course).
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.id
            FROM exams e
            JOIN subjects s ON s.id = e.subject_id
            WHERE s.course_id = %s
            ORDER BY e.created_at DESC
            LIMIT 1
            """,
            (course_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def find_students_by_last3(*, batch_id: int | None, course_id: int, prn_last3: str):
    prn_last3 = (prn_last3 or "").strip()
    if len(prn_last3) != 3 or not prn_last3.isdigit():
        return []

    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # If batch_id is available, scope it too (safest)
        if batch_id is not None:
            cur.execute(
                """
                SELECT prn, name
                FROM students
                WHERE batch_id=%s AND course_id=%s AND prn_last3=%s
                ORDER BY prn
                """,
                (batch_id, course_id, prn_last3),
            )
        else:
            # fallback if batch_id missing
            cur.execute(
                """
                SELECT prn, name
                FROM students
                WHERE course_id=%s AND prn_last3=%s
                ORDER BY prn
                """,
                (course_id, prn_last3),
            )

        return cur.fetchall() or []
    finally:
        conn.close()



def change_student_password(prn: str, new_password: str) -> bool:
    """
    Student changes password.
    """
    prn = str(prn).strip()
    if not prn or not new_password:
        return False

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE students SET password_hash=%s WHERE prn=%s",
            (_hash_password(new_password), prn),
        )
        return cur.rowcount > 0
    finally:
        conn.close()


def admin_reset_student_password_to_prn(prn: str) -> bool:
    """
    Admin resets password back to PRN.
    """
    prn = str(prn).strip()
    if not prn:
        return False

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE students SET password_hash=%s WHERE prn=%s",
            (_hash_password(prn), prn),
        )
        return cur.rowcount > 0
    finally:
        conn.close()


def list_students_for_course(course_id: int):
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT id, prn, name, phone, email, batch_id, course_id, created_at, updated_at
            FROM students
            WHERE course_id=%s
            ORDER BY prn
            """,
            (course_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def update_student_contact(prn: str, phone: str | None, email: str | None) -> bool:
    """
    Used by student portal (student can update phone/email).
    """
    prn = str(prn).strip()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE students
            SET phone=%s, email=%s
            WHERE prn=%s
            """,
            (phone, email, prn),
        )
        return cur.rowcount > 0
    finally:
        conn.close()

def auto_link_exam_students_by_last3(exam_id: int) -> dict:
    """
    For a given exam_id:
    - find course_id, batch_id via exam -> subject -> course
    - for each exam_student, match by last3 within (batch_id, course_id)
    - if exactly 1 match, update exam_students prn+name, set is_conflict=0
    """
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # get subject_id for exam
        cur.execute("SELECT subject_id FROM exams WHERE id=%s", (exam_id,))
        r = cur.fetchone()
        if not r:
            return {"updated": 0, "conflicts": 0, "no_match": 0}

        subject_id = int(r["subject_id"])

        # get course_id from subject
        cur.execute("SELECT course_id FROM subjects WHERE id=%s", (subject_id,))
        r = cur.fetchone()
        if not r:
            return {"updated": 0, "conflicts": 0, "no_match": 0}
        course_id = int(r["course_id"])

        # get batch_id from course
        cur.execute("SELECT batch_id FROM courses WHERE id=%s", (course_id,))
        r = cur.fetchone()
        batch_id = int(r["batch_id"]) if r and r["batch_id"] is not None else None

        # fetch exam_students
        cur.execute("SELECT id, prn FROM exam_students WHERE exam_id=%s", (exam_id,))
        exam_rows = cur.fetchall() or []

        updated = 0
        conflicts = 0
        no_match = 0

        for row in exam_rows:
            es_id = int(row["id"])
            prn = str(row["prn"] or "")

            digits = "".join(ch for ch in prn if ch.isdigit())
            if not digits:
                no_match += 1
                continue

            # ✅ if OCR dropped leading zeros (e.g., "21"), treat it as "021"
            last3 = digits[-3:] if len(digits) >= 3 else digits.zfill(3)




            # match in students table (scoped)
            if batch_id is not None:
                cur.execute(
                    """
                    SELECT prn, name
                    FROM students
                    WHERE batch_id=%s AND course_id=%s AND prn_last3=%s
                    """,
                    (batch_id, course_id, last3),
                )
            else:
                cur.execute(
                    """
                    SELECT prn, name
                    FROM students
                    WHERE course_id=%s AND prn_last3=%s
                    """,
                    (course_id, last3),
                )

            matches = cur.fetchall() or []
            if len(matches) == 1:
                m = matches[0]
                cur2 = conn.cursor()
                cur2.execute(
                    """
                    UPDATE exam_students
                    SET prn=%s, name=%s, is_conflict=0
                    WHERE id=%s
                    """,
                    (m["prn"], m["name"], es_id),
                )
                updated += 1
            elif len(matches) == 0:
                no_match += 1
            else:
                conflicts += 1

        return {"updated": updated, "conflicts": conflicts, "no_match": no_match}

    finally:
        conn.close()

def save_exam_results(subject_id: int, pdf_path: str, key_path: str, students: dict) -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()

        # create exam
        cur.execute(
            """
            INSERT INTO exams (subject_id, pdf_path, key_path)
            VALUES (%s, %s, %s)
            """,
            (subject_id, pdf_path, key_path),
        )
        exam_id = cur.lastrowid

        # insert students + answers
        for prn, entries in students.items():
            is_conflict = 1 if len(entries) > 1 else 0

            for entry in entries:
                cur.execute(
                    """
                    INSERT INTO exam_students
                        (exam_id, prn, name, score, total_questions, image_path, is_conflict)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        exam_id,
                        prn,
                        entry.get("name", "Unknown"),
                        int(entry.get("score", 0)),
                        int(entry.get("total", 0)),
                        entry.get("image_path"),
                        is_conflict,
                    ),
                )

                exam_student_id = cur.lastrowid

                answer_rows = []
                for d in entry.get("details", []):
                    student_answer = d["student_answer"]
                    if student_answer == "(blank)":
                        student_answer = None

                    answer_rows.append(
                        (
                            exam_student_id,
                            int(d["question"]),
                            student_answer,
                            d["key_answer"],
                            int(d["is_correct"]),
                            int(d["is_blank"]),
                        )
                    )

                if answer_rows:
                    cur.executemany(
                        """
                        INSERT INTO exam_answers
                            (exam_student_id, question_no, student_answer,
                             key_answer, is_correct, is_blank)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        answer_rows,
                    )

        return exam_id
    finally:
        conn.close()



def load_exam_results(subject_id: int):
    """
    Load the latest exam for a subject.
    Returns a dict:
    {
        "exam_id": ...,
        "pdf_path": ...,
        "key_path": ...,
        "answer_key": {q_no: correct_option, ...},
        "students": {
            prn: {
                "name": ...,
                "prn": ...,
                "score": ...,
                "total": ...,
                "image_path": ...,
                "details": [ {question, student_answer, key_answer, is_correct, is_blank}, ... ]
            },
            ...
        }
    }
    or None if no exam exists.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # latest exam for this subject
        cur.execute(
            """
            SELECT id, pdf_path, key_path
            FROM exams
            WHERE subject_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (subject_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        exam_id, pdf_path, key_path = row

        # get all students for this exam
        cur.execute(
            """
            SELECT id, prn, name, score, total_questions, image_path
            FROM exam_students
            WHERE exam_id = %s
            """,
            (exam_id,),
        )
        students_rows = cur.fetchall()

        key_map = {}
        students = {}

        for exam_student_id, prn, name, score, total, image_path in students_rows:
            cur_answers = conn.cursor()
            cur_answers.execute(
                """
                SELECT question_no, student_answer, key_answer, is_correct, is_blank
                FROM exam_answers
                WHERE exam_student_id = %s
                ORDER BY question_no
                """,
                (exam_student_id,),
            )

            details = []
            for q_no, s_ans, k_ans, is_corr, is_blank in cur_answers.fetchall():
                k_ans = (k_ans or "").strip().upper()
                if q_no not in key_map:
                    key_map[q_no] = k_ans

                student_answer = s_ans if s_ans else "(blank)"

                details.append(
                    {
                        "question": int(q_no),
                        "student_answer": student_answer,
                        "key_answer": k_ans,
                        "is_correct": bool(is_corr),
                        "is_blank": bool(is_blank),
                    }
                )

            if prn not in students:
                students[prn] = []

            students[prn].append(
                {
                    "exam_student_id": exam_student_id,
                    "name": name,
                    "prn": prn,
                    "score": int(score),
                    "total": int(total),
                    "image_path": image_path,
                    "details": details,
                }
            )

        return {
            "exam_id": exam_id,
            "pdf_path": pdf_path,
            "key_path": key_path,
            "answer_key": key_map,
            "students": students,
        }
    finally:
        conn.close()



def update_exam_student_answers(exam_student_id: int, score: int, details: list):
    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute(
            "UPDATE exam_students SET score = %s WHERE id = %s",
            (score, exam_student_id),
        )

        cur.execute(
            "DELETE FROM exam_answers WHERE exam_student_id = %s",
            (exam_student_id,),
        )

        answer_rows = []
        for d in details:
            student_answer = d["student_answer"]
            if student_answer == "(blank)":
                student_answer = None

            answer_rows.append(
                (
                    exam_student_id,
                    int(d["question"]),
                    student_answer,
                    d["key_answer"],
                    int(d["is_correct"]),
                    int(d["is_blank"]),
                )
            )

        if answer_rows:
            cur.executemany(
                """
                INSERT INTO exam_answers
                    (exam_student_id, question_no, student_answer,
                     key_answer, is_correct, is_blank)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                answer_rows,
            )
    finally:
        conn.close()


def update_exam_student_identity(exam_student_id: int, new_prn: str, new_name: str):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE exam_students
            SET prn = %s, name = %s, is_conflict = 0
            WHERE id = %s
            """,
            (new_prn, new_name, exam_student_id),
        )
    finally:
        conn.close()

def get_latest_exam_id_for_subject(subject_id: int) -> int | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id
            FROM exams
            WHERE subject_id=%s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (subject_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def publish_latest_exam_for_subject(subject_id: int, admin_id: int | None = None) -> dict:
    """
    Publishes the LATEST exam of this subject to student portal.
    If no exam exists, returns error.
    """
    exam_id = get_latest_exam_id_for_subject(subject_id)
    if not exam_id:
        return {"ok": False, "error": "No exam found for this subject to publish."}

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO subject_publish (subject_id, exam_id, is_published, published_by, published_at)
            VALUES (%s, %s, 1, %s, NOW())
            ON DUPLICATE KEY UPDATE
                exam_id=VALUES(exam_id),
                is_published=1,
                published_by=VALUES(published_by),
                published_at=NOW()
            """,
            (subject_id, exam_id, admin_id),
        )
        return {"ok": True, "subject_id": subject_id, "exam_id": exam_id}
    finally:
        conn.close()


def unpublish_subject(subject_id: int) -> bool:
    """
    Stops showing this subject in student portal (even if exam exists).
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE subject_publish SET is_published=0 WHERE subject_id=%s",
            (subject_id,),
        )
        return cur.rowcount > 0
    finally:
        conn.close()


def is_subject_published(subject_id: int) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT is_published
            FROM subject_publish
            WHERE subject_id=%s
            """,
            (subject_id,),
        )
        row = cur.fetchone()
        return bool(row[0]) if row else False
    finally:
        conn.close()


def get_published_exam_id(subject_id: int) -> int | None:
    """
    Returns the published exam_id if subject is published, else None.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT exam_id
            FROM subject_publish
            WHERE subject_id=%s AND is_published=1
            """,
            (subject_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()

def get_student_exam_sheet_and_key(*, prn: str, exam_id: int) -> dict | None:
    """
    For a PUBLISHED exam:
    Returns:
      {
        "exam_student_id": int,
        "image_path": str|None,
        "score": int,
        "total_questions": int,
        "answers": [
            {question_no, student_answer, key_answer, is_correct, is_blank}
        ]
      }
    """
    prn = _normalize_prn(prn)
    if not prn or not exam_id:
        return None

    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # get student's exam row (latest if duplicates)
        cur.execute(
            """
            SELECT id AS exam_student_id, image_path, score, total_questions
            FROM exam_students
            WHERE exam_id=%s AND prn=%s
            ORDER BY id DESC
            LIMIT 1
            """,
            (exam_id, prn),
        )
        es = cur.fetchone()
        if not es:
            return None

        exam_student_id = int(es["exam_student_id"])

        # get answers (question_no + key + student)
        cur.execute(
            """
            SELECT question_no, student_answer, key_answer, is_correct, is_blank
            FROM exam_answers
            WHERE exam_student_id=%s
            ORDER BY question_no
            """,
            (exam_student_id,),
        )
        rows = cur.fetchall() or []

        # normalize
        for r in rows:
            if not r.get("student_answer"):
                r["student_answer"] = "(blank)"
            if r.get("key_answer"):
                r["key_answer"] = str(r["key_answer"]).strip().upper()

        return {
            "exam_student_id": exam_student_id,
            "image_path": es.get("image_path"),
            "score": int(es.get("score") or 0),
            "total_questions": int(es.get("total_questions") or 0),
            "answers": rows,
        }
    finally:
        conn.close()


# ------------------------------------------------------------------
# Question Paper + MCQ Bank (LLM) operations (NEW)
# ------------------------------------------------------------------

def save_question_paper_upload(subject_id: int, qp_pdf_path: str, answer_key_path: str | None = None) -> int:
    """
    Creates a new question paper "version" for a subject.
    Returns question_paper_id.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO question_papers (subject_id, qp_pdf_path, answer_key_path)
            VALUES (%s, %s, %s)
            """,
            (subject_id, qp_pdf_path, answer_key_path),
        )
        return int(cur.lastrowid)
    finally:
        conn.close()


def load_latest_question_paper(subject_id: int):
    """
    Returns latest question_paper row dict: {id, qp_pdf_path, answer_key_path, created_at}
    or None.
    """
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT id, qp_pdf_path, answer_key_path, created_at
            FROM question_papers
            WHERE subject_id=%s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (subject_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def save_mcq_bank_items(question_paper_id: int, items: list[dict]) -> int:
    """
    Bulk insert MCQs produced by LLM.
    Each item dict expected keys:
      question_no, question_text, option_a, option_b, option_c, option_d,
      correct_option, why_correct, why_a_wrong, why_b_wrong, why_c_wrong, why_d_wrong
    Returns inserted_count.
    """
    if not items:
        return 0

    rows = []
    for it in items:
        qno = int(it.get("question_no", 0) or 0)
        if qno <= 0:
            continue

        corr = (it.get("correct_option") or "").strip().upper()
        corr = corr[:1] if corr else None

        rows.append(
            (
                question_paper_id,
                qno,
                it.get("question_text"),
                it.get("option_a"),
                it.get("option_b"),
                it.get("option_c"),
                it.get("option_d"),
                corr,
                it.get("why_correct"),
                it.get("why_a_wrong"),
                it.get("why_b_wrong"),
                it.get("why_c_wrong"),
                it.get("why_d_wrong"),
            )
        )

    if not rows:
        return 0

    conn = get_connection()
    try:
        cur = conn.cursor()

        # If you re-upload / re-run LLM for same question_paper_id, clear previous rows
        cur.execute("DELETE FROM mcq_bank WHERE question_paper_id=%s", (question_paper_id,))

        cur.executemany(
            """
            INSERT INTO mcq_bank (
                question_paper_id, question_no, question_text,
                option_a, option_b, option_c, option_d,
                correct_option,
                why_correct, why_a_wrong, why_b_wrong, why_c_wrong, why_d_wrong
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            rows,
        )
        return int(cur.rowcount)
    finally:
        conn.close()


def load_mcq_bank_for_subject(subject_id: int, *, latest: bool = True):
    """
    Returns:
      {
        "question_paper": {...},
        "questions": [ {question_no, question_text, option_a..d, correct_option, why_*}, ... ]
      }
    or None if no question paper exists.
    """
    qp = load_latest_question_paper(subject_id) if latest else None
    if not qp:
        return None

    qp_id = int(qp["id"])

    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT
                question_no, question_text,
                option_a, option_b, option_c, option_d,
                correct_option,
                why_correct, why_a_wrong, why_b_wrong, why_c_wrong, why_d_wrong
            FROM mcq_bank
            WHERE question_paper_id=%s
            ORDER BY question_no
            """,
            (qp_id,),
        )
        return {"question_paper": qp, "questions": cur.fetchall() or []}
    finally:
        conn.close()


def get_student_question_review(prn: str, subject_id: int, exam_id: int | None = None):
    """
    Student portal view (PUBLISHED-AWARE):
    - If exam_id is provided -> uses that exam (published exam)
    - Else -> uses latest exam for subject
    - Uses latest question paper for subject
    Returns rows:
      question_no, question_text, options, correct_option, explanations,
      student_answer, key_answer, is_correct, is_blank
    """
    prn = _normalize_prn(prn)
    if not prn:
        return []

    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # 1) decide which exam to use
        if not exam_id:
            cur.execute(
                """
                SELECT id AS exam_id
                FROM exams
                WHERE subject_id=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (subject_id,),
            )
            ex = cur.fetchone()
            if not ex:
                return []
            exam_id = int(ex["exam_id"])

        # 2) exam_student row for this PRN (latest if duplicates)
        cur.execute(
            """
            SELECT id AS exam_student_id
            FROM exam_students
            WHERE exam_id=%s AND prn=%s
            ORDER BY id DESC
            LIMIT 1
            """,
            (exam_id, prn),
        )
        es = cur.fetchone()
        if not es:
            return []

        exam_student_id = int(es["exam_student_id"])

        # 3) latest question paper for subject
        qp = load_latest_question_paper(subject_id)
        if not qp:
            # return answers even if mcq text isn't present
            cur.execute(
                """
                SELECT question_no, student_answer, key_answer, is_correct, is_blank
                FROM exam_answers
                WHERE exam_student_id=%s
                ORDER BY question_no
                """,
                (exam_student_id,),
            )
            rows = cur.fetchall() or []
            for r in rows:
                if not r.get("student_answer"):
                    r["student_answer"] = "(blank)"
                if r.get("key_answer"):
                    r["key_answer"] = str(r["key_answer"]).strip().upper()
            return rows

        qp_id = int(qp["id"])

        # 4) join: exam_answers + mcq_bank
        cur.execute(
            """
            SELECT
                ea.question_no,

                mb.question_text,
                mb.option_a, mb.option_b, mb.option_c, mb.option_d,
                mb.correct_option,
                mb.why_correct, mb.why_a_wrong, mb.why_b_wrong, mb.why_c_wrong, mb.why_d_wrong,

                ea.student_answer,
                ea.key_answer,
                ea.is_correct,
                ea.is_blank
            FROM exam_answers ea
            LEFT JOIN mcq_bank mb
              ON mb.question_paper_id = %s
             AND mb.question_no = ea.question_no
            WHERE ea.exam_student_id = %s
            ORDER BY ea.question_no
            """,
            (qp_id, exam_student_id),
        )

        rows = cur.fetchall() or []

        # normalize
        for r in rows:
            if not r.get("student_answer"):
                r["student_answer"] = "(blank)"
            if r.get("correct_option"):
                r["correct_option"] = str(r["correct_option"]).strip().upper()
            if r.get("key_answer"):
                r["key_answer"] = str(r["key_answer"]).strip().upper()

        return rows
    finally:
        conn.close()
# ------------------------------------------------------------------
# Lab Marks (per subject)
# ------------------------------------------------------------------

def upsert_lab_marks(subject_id: int, prn: str, marks, updated_by: int | None = None) -> bool:
    """Insert or update a single student's lab marks for a subject."""
    #prn = (prn or "").strip().upper()
    prn = _normalize_prn(prn)


    # marks can be blank -> NULL
    marks_val = None
    try:
        if marks is not None and str(marks).strip() != "":
            marks_val = float(marks)
    except Exception:
        marks_val = None

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO lab_marks (subject_id, prn, marks, updated_by)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                marks = VALUES(marks),
                updated_by = VALUES(updated_by),
                updated_at = CURRENT_TIMESTAMP
            """,
            (subject_id, prn, marks_val, updated_by),
        )
        conn.commit()
        return True
    except Error as e:
        print("DB upsert_lab_marks error:", e)
        return False
    finally:
        cursor.close()
        conn.close()


def import_lab_marks_from_excel(subject_id: int, excel_file, updated_by: int | None = None) -> dict:
    """
    Bulk import lab marks for a subject from uploaded Excel.
    Expected: first column PRN, second column LAB_MARKS (or any name).
    """
    if excel_file is None:
        return {"inserted_or_updated": 0, "skipped": 0}

    df = pd.read_excel(excel_file)
    if df.shape[1] < 2:
        return {"inserted_or_updated": 0, "skipped": len(df)}

    prn_col = df.columns[0]
    mark_col = df.columns[1]

    rows = []
    skipped = 0

    for _, r in df.iterrows():
        prn = r.get(prn_col, None)
        marks = r.get(mark_col, None)

        if pd.isna(prn):
            skipped += 1
            continue

        #prn = str(prn).strip().upper()
        prn = _normalize_prn(prn)

        if not prn:
            skipped += 1
            continue

        marks_val = None
        try:
            if marks is not None and not pd.isna(marks) and str(marks).strip() != "":
                marks_val = float(marks)
        except Exception:
            marks_val = None

        rows.append((subject_id, prn, marks_val, updated_by))

    if not rows:
        return {"inserted_or_updated": 0, "skipped": skipped}

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany(
            """
            INSERT INTO lab_marks (subject_id, prn, marks, updated_by)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                marks = VALUES(marks),
                updated_by = VALUES(updated_by),
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()
        return {"inserted_or_updated": cursor.rowcount, "skipped": skipped}
    except Error as e:
        print("DB import_lab_marks_from_excel error:", e)
        return {"inserted_or_updated": 0, "skipped": skipped}
    finally:
        cursor.close()
        conn.close()


def get_lab_marks_for_subject(subject_id: int, course_id: int | None = None) -> list[dict]:
    """
    Returns rows for Lab marks screen.
    If course_id is given: returns ALL students in the course, with lab marks if present.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        if course_id is None:
            cursor.execute(
                """
                SELECT prn, marks, updated_at
                FROM lab_marks
                WHERE subject_id = %s
                ORDER BY prn
                """,
                (subject_id,),
            )
        else:
            cursor.execute(
                """
                SELECT s.prn, s.name, lm.marks, lm.updated_at
                FROM students s
                LEFT JOIN lab_marks lm
                    ON lm.subject_id = %s AND TRIM(lm.prn) = TRIM(s.prn)
                WHERE s.course_id = %s
                ORDER BY s.prn
                """,
                (subject_id, course_id),
            )
        return cursor.fetchall()
    except Error as e:
        print("DB get_lab_marks_for_subject error:", e)
        return []
    finally:
        cursor.close()
        conn.close()


def get_lab_marks_map(subject_id: int) -> dict[str, float | None]:
    """Convenience map: {PRN: marks} for report generation."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT prn, marks
            FROM lab_marks
            WHERE subject_id = %s
            """,
            (subject_id,),
        )
        out: dict[str, float | None] = {}
        for prn, marks in cursor.fetchall():
            #prn_norm = str(prn).strip().upper()
            prn_norm = _normalize_prn(prn)

            out[prn_norm] = float(marks) if marks is not None else None
        return out
    except Error as e:
        print("DB get_lab_marks_map error:", e)
        return {}
    finally:
        cursor.close()
        conn.close()


def get_course_report(course_id: int, pass_percent: float = 35.0) -> dict:
    """
    Course-wide report across all subjects using:
      - students table => enrolled PRNs (for absent/present)
      - latest exam per subject => marks stats

    pass_percent: percent threshold to consider pass/fail (default 35%)
    """
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # course + batch
        cur.execute(
            """
            SELECT c.id AS course_id, c.name AS course_name,
                   b.id AS batch_id, b.name AS batch_name
            FROM courses c
            LEFT JOIN batches b ON b.id = c.batch_id
            WHERE c.id = %s
            LIMIT 1
            """,
            (course_id,),
        )
        course_row = cur.fetchone()
        if not course_row:
            return {"error": f"Course not found: {course_id}"}

        # enrolled students in this course (PRN list)
        cur.execute(
            """
            SELECT prn, name
            FROM students
            WHERE course_id = %s
            ORDER BY prn
            """,
            (course_id,),
        )
        enrolled = cur.fetchall() or []
        enrolled_prns = {str(r["prn"]).strip(): (r.get("name") or "") for r in enrolled}
        total_enrolled = len(enrolled_prns)

        # subjects for course
        cur.execute(
            """
            SELECT id AS subject_id, name AS subject_name
            FROM subjects
            WHERE course_id = %s
            ORDER BY id
            """,
            (course_id,),
        )
        subjects = cur.fetchall() or []

        subject_reports = []
        # overall aggregation across latest exams per subject
        overall_by_prn = {}  # prn -> {"name":.., "score_sum":.., "total_sum":.., "attempted_subjects":..}

        for s in subjects:
            subject_id = int(s["subject_id"])
            subject_name = s["subject_name"]

            # latest exam for this subject
            cur.execute(
                """
                SELECT id, created_at
                FROM exams
                WHERE subject_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (subject_id,),
            )
            ex = cur.fetchone()
            if not ex:
                subject_reports.append({
                    "subject_id": subject_id,
                    "subject_name": subject_name,
                    "has_exam": False,
                    "message": "No exam uploaded yet for this subject."
                })
                continue

            exam_id = int(ex["id"])

            # marks for this exam
            cur.execute(
                """
                SELECT prn, name, score, total_questions
                FROM exam_students
                WHERE exam_id = %s
                """,
                (exam_id,),
            )
            rows = cur.fetchall() or []

            present_prns = set()
            scores = []
            topper_score = None
            toppers = []
            fail_count = 0

            for r in rows:
                prn = str(r["prn"]).strip()
                name = (r.get("name") or enrolled_prns.get(prn) or "").strip()
                score = int(r.get("score") or 0)
                total_q = int(r.get("total_questions") or 0)

                present_prns.add(prn)
                scores.append(score)

                pct = (score / total_q * 100.0) if total_q > 0 else 0.0
                if pct < pass_percent:
                    fail_count += 1

                # top
                if topper_score is None or score > topper_score:
                    topper_score = score
                    toppers = [{"prn": prn, "name": name, "score": score, "percent": pct}]
                elif score == topper_score:
                    toppers.append({"prn": prn, "name": name, "score": score, "percent": pct})

                # overall aggregation
                if prn not in overall_by_prn:
                    overall_by_prn[prn] = {
                        "prn": prn,
                        "name": name or enrolled_prns.get(prn, ""),
                        "score_sum": 0,
                        "total_sum": 0,
                        "attempted_subjects": 0,
                    }
                overall_by_prn[prn]["score_sum"] += score
                overall_by_prn[prn]["total_sum"] += total_q
                overall_by_prn[prn]["attempted_subjects"] += 1

            present_count = len(present_prns)
            absent_count = max(0, total_enrolled - present_count)

            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
            else:
                avg_score = 0.0
                min_score = 0
                max_score = 0

            fail_rate = (fail_count / present_count * 100.0) if present_count > 0 else 0.0

            subject_reports.append({
                "subject_id": subject_id,
                "subject_name": subject_name,
                "has_exam": True,
                "latest_exam_id": exam_id,
                "enrolled": total_enrolled,
                "present": present_count,
                "absent": absent_count,
                "min": min_score,
                "max": max_score,
                "avg": round(avg_score, 2),
                "toppers": toppers,          # list (handles ties)
                "fail_count": fail_count,
                "fail_rate_percent": round(fail_rate, 2),
                "pass_percent_threshold": pass_percent,
            })

        # Overall toppers (based on sum across latest exams of subjects attempted)
        overall_list = list(overall_by_prn.values())
        for r in overall_list:
            total_sum = r["total_sum"]
            r["overall_percent"] = round((r["score_sum"] / total_sum * 100.0), 2) if total_sum > 0 else 0.0

        overall_list.sort(key=lambda x: (x["score_sum"], x["overall_percent"]), reverse=True)
        overall_top = overall_list[:10]  # top 10

        # hardest subjects (highest fail rate)
        hardest = [x for x in subject_reports if x.get("has_exam")]
        hardest.sort(key=lambda x: x.get("fail_rate_percent", 0), reverse=True)

        return {
            "course": course_row,
            "total_enrolled": total_enrolled,
            "subjects_total": len(subjects),
            "subjects_with_exam": sum(1 for x in subject_reports if x.get("has_exam")),
            "pass_percent_threshold": pass_percent,
            "subject_reports": subject_reports,
            "overall_topper_list": overall_top,
            "hardest_subjects_by_fail_rate": hardest[:5],
        }

    except Error as e:
        return {"error": f"DB error: {e}"}
    finally:
        conn.close()
def get_student_wise_report(
    *,
    prn: str,
    course_id: int,
    theory_max: float = 40.0,
    lab_max: float = 40.0,
    pass_mark: float = 16.0,
) -> dict:
    """
    Student report across all subjects in a course.

    LAB logic:
    - If NO lab rows exist at all for that subject => Lab="NA"
    - Else (lab exists for that subject):
        - If this student's lab marks is NULL or missing => Lab="AB"
        - Else => numeric mark
    """

    prn = _normalize_prn(prn)
    if not prn:
        return {"error": "Invalid PRN"}

    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # student + course + batch
        cur.execute(
            """
            SELECT
                s.prn, s.name,
                s.course_id, c.name AS course_name,
                s.batch_id, b.name AS batch_name
            FROM students s
            LEFT JOIN courses c ON c.id = s.course_id
            LEFT JOIN batches b ON b.id = s.batch_id
            WHERE s.prn=%s AND s.course_id=%s
            LIMIT 1
            """,
            (prn, course_id),
        )
        student = cur.fetchone()
        if not student:
            return {"error": f"Student not found in course. PRN={prn}, course_id={course_id}"}

        # subjects in course
        cur.execute(
            """
            SELECT id AS subject_id, name AS subject_name
            FROM subjects
            WHERE course_id=%s
            ORDER BY id
            """,
            (course_id,),
        )
        subjects = cur.fetchall() or []

        out_rows = []

        total_theory = 0.0
        total_lab_numeric = 0.0
        overall_total = 0.0
        overall_possible = 0.0

        failed_subjects = []
        below_20_subjects = []

        for s in subjects:
            subject_id = int(s["subject_id"])
            subject_name = s["subject_name"]

            # -----------------------
            # THEORY: latest exam?
            # -----------------------
            cur.execute(
                """
                SELECT id
                FROM exams
                WHERE subject_id=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (subject_id,),
            )
            ex = cur.fetchone()
            exam_uploaded = bool(ex)

            theory_mark = None
            class_scores = []
            rank = None

            if exam_uploaded:
                exam_id = int(ex["id"])

                # all scores for ranking + class stats
                cur.execute(
                    """
                    SELECT prn, score
                    FROM exam_students
                    WHERE exam_id=%s
                    """,
                    (exam_id,),
                )
                all_rows = cur.fetchall() or []
                for r in all_rows:
                    class_scores.append(int(r["score"] or 0))

                # student theory score (if absent from exam_students => treat as 0)
                cur.execute(
                    """
                    SELECT score
                    FROM exam_students
                    WHERE exam_id=%s AND prn=%s
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (exam_id, prn),
                )
                me = cur.fetchone()
                theory_mark = int(me["score"]) if me else 0

                # rank (1 = highest). If tie: same rank style by counting > score.
                if class_scores:
                    higher = sum(1 for x in class_scores if x > theory_mark)
                    rank = higher + 1

            # class stats
            if class_scores:
                class_avg = round(sum(class_scores) / len(class_scores), 2)
                class_min = min(class_scores)
                class_max = max(class_scores)
            else:
                class_avg, class_min, class_max = 0.0, 0, 0

            # -----------------------
            # LAB: NA / AB / numeric
            # -----------------------
            # lab_applicable = any row exists for this subject in lab_marks
            cur.execute(
                """
                SELECT 1
                FROM lab_marks
                WHERE subject_id=%s
                LIMIT 1
                """,
                (subject_id,),
            )
            lab_applicable = cur.fetchone() is not None

            lab_mark_display = "NA"
            lab_mark_numeric = 0.0

            if lab_applicable:
                cur.execute(
                    """
                    SELECT marks
                    FROM lab_marks
                    WHERE subject_id=%s AND prn=%s
                    LIMIT 1
                    """,
                    (subject_id, prn),
                )
                lm = cur.fetchone()
                if not lm or lm["marks"] is None:
                    lab_mark_display = "AB"
                else:
                    lab_mark_numeric = float(lm["marks"])
                    lab_mark_display = round(lab_mark_numeric, 2)

            # -----------------------
            # TOTAL + STATUS
            # -----------------------
            t = float(theory_mark or 0)
            total_theory += t

            total_possible = float(theory_max) + (float(lab_max) if lab_applicable else 0.0)
            total = t + (lab_mark_numeric if lab_applicable else 0.0)

            # percent only on applicable total_possible
            percent = round((total / total_possible * 100.0), 2) if total_possible > 0 else 0.0

            # Status rules:
            # - If no exam uploaded -> "NO EXAM"
            # - If theory < pass_mark -> FAIL
            # - If lab NA -> don't fail on lab
            # - If lab AB -> ABSENT
            # - If lab numeric < pass_mark -> FAIL
            # else PASS
            if not exam_uploaded:
                status = "NO EXAM"
            else:
                if t < pass_mark:
                    status = "FAIL"
                else:
                    if not lab_applicable:
                        status = "PASS"
                    else:
                        if lab_mark_display == "AB":
                            status = "ABSENT"
                        else:
                            if float(lab_mark_numeric) < pass_mark:
                                status = "FAIL"
                            else:
                                status = "PASS"

            # subject flags
            if status == "FAIL":
                failed_subjects.append(subject_name)

            if exam_uploaded:
                # "study hard" based on total < 20 (only if exam exists; lab NA counts as 0)
                if total < 20:
                    below_20_subjects.append(subject_name)

            # overall totals: count only theory if exam exists; lab only if lab_applicable
            # (You can change this if you want to include even when NO EXAM)
            if exam_uploaded:
                overall_total += total
                overall_possible += total_possible
                total_lab_numeric += (lab_mark_numeric if lab_applicable else 0.0)

            out_rows.append(
                {
                    "subject_id": subject_id,
                    "subject_name": subject_name,
                    "exam_uploaded": "Yes" if exam_uploaded else "No",
                    "theory_mark": t if exam_uploaded else "",
                    "lab_mark": lab_mark_display,
                    "total": round(total, 2) if exam_uploaded else "",
                    "total_possible": total_possible if exam_uploaded else "",
                    "percent": percent if exam_uploaded else "",
                    "status": status,
                    "rank": rank if exam_uploaded else "",
                    "class_avg_theory": class_avg,
                    "class_min_theory": class_min,
                    "class_max_theory": class_max,
                }
            )

        overall_percent = round((overall_total / overall_possible * 100.0), 2) if overall_possible > 0 else 0.0

        return {
            "student": student,
            "subjects": out_rows,
            "summary": {
                "total_theory": round(total_theory, 2),
                "total_lab": round(total_lab_numeric, 2),
                "overall_total": round(overall_total, 2),
                "overall_possible": round(overall_possible, 2),
                "overall_percent": overall_percent,
                "failed_subjects": failed_subjects,
                "below_20_subjects": below_20_subjects,
            },
        }

    except Error as e:
        return {"error": f"DB error: {e}"}
    finally:
        conn.close()
def get_student_report_published_only(
    *,
    prn: str,
    course_id: int,
    theory_max: float = 40.0,
    lab_max: float = 40.0,
    pass_total_min: float = 16.0,
) -> dict:
    """
    Student-side report:
    - ONLY subjects that are published
    - per subject: theory (scaled to 40), lab (0-40 or NULL), total, pass/fail, class avg, rank
    """

    prn = _normalize_prn(prn)
    if not prn or not course_id:
        return {"rows": [], "summary": {}}

    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # 1) subjects for course
        cur.execute(
            """
            SELECT id AS subject_id, name AS subject_name
            FROM subjects
            WHERE course_id=%s
            ORDER BY id
            """,
            (course_id,),
        )
        all_subjects = cur.fetchall() or []

        # 2) keep ONLY published
        published_subjects = []
        for s in all_subjects:
            sid = int(s["subject_id"])
            if is_subject_published(sid):
                published_subjects.append(s)

        if not published_subjects:
            return {"rows": [], "summary": {"message": "No published subjects yet."}}

        rows_out = []

        # helper: lab marks for this student across subjects
        cur.execute(
            """
            SELECT subject_id, marks
            FROM lab_marks
            WHERE prn=%s
            """,
            (prn,),
        )
        lab_map = {int(r["subject_id"]): r["marks"] for r in (cur.fetchall() or [])}

        for s in published_subjects:
            subject_id = int(s["subject_id"])
            subject_name = s["subject_name"]

            exam_id = get_published_exam_id(subject_id)
            if not exam_id:
                # published subject but exam_id missing → treat as not available
                rows_out.append(
                    {
                        "subject_id": subject_id,
                        "subject_name": subject_name,
                        "theory_marks": None,
                        "lab_marks": lab_map.get(subject_id, None),
                        "total": None,
                        "status": "Not available",
                        "class_avg_total": None,
                        "rank_total": None,
                        "class_size": None,
                    }
                )
                continue

            # ---- student exam record (theory raw score + total_questions) ----
            cur.execute(
                """
                SELECT id AS exam_student_id, score, total_questions
                FROM exam_students
                WHERE exam_id=%s AND prn=%s
                ORDER BY id DESC
                LIMIT 1
                """,
                (exam_id, prn),
            )
            es = cur.fetchone()

            if not es:
                # absent in theory exam
                theory_marks = None
            else:
                score = float(es.get("score") or 0)
                total_q = float(es.get("total_questions") or 0)
                theory_marks = round((score / total_q) * theory_max, 2) if total_q > 0 else 0.0

            lab_marks = lab_map.get(subject_id, None)
            lab_marks = float(lab_marks) if lab_marks is not None else None

            # total only if at least something exists
            if theory_marks is None and lab_marks is None:
                total = None
            else:
                total = round((theory_marks or 0.0) + (lab_marks or 0.0), 2)

            status = "Not available"
            if total is not None:
                status = "PASS" if total >= pass_total_min else "FAIL"

            # ---- class average + rank (based on TOTAL = theory_scaled + lab) ----
            # get all students totals for this subject (for published exam)
            cur.execute(
                """
                SELECT prn, score, total_questions
                FROM exam_students
                WHERE exam_id=%s
                """,
                (exam_id,),
            )
            class_exam = cur.fetchall() or []

            # lab marks for all students in this subject
            cur.execute(
                """
                SELECT prn, marks
                FROM lab_marks
                WHERE subject_id=%s
                """,
                (subject_id,),
            )
            class_lab = { _normalize_prn(r["prn"]): r["marks"] for r in (cur.fetchall() or [])}

            totals = []  # list of (prn, total)
            for r in class_exam:
                p = _normalize_prn(r["prn"])
                score = float(r.get("score") or 0)
                tq = float(r.get("total_questions") or 0)
                th = round((score / tq) * theory_max, 2) if tq > 0 else 0.0
                lb = class_lab.get(p, None)
                lb = float(lb) if lb is not None else 0.0
                totals.append((p, round(th + lb, 2)))

            # if exam exists but totals empty
            if totals:
                class_size = len(totals)
                avg_total = round(sum(t for _, t in totals) / class_size, 2)

                # rank: higher total = better rank, handle ties (dense rank)
                totals_sorted = sorted(totals, key=lambda x: x[1], reverse=True)
                rank = None
                current_rank = 0
                last_score = None
                for idx, (p, t) in enumerate(totals_sorted):
                    if last_score is None or t != last_score:
                        current_rank = current_rank + 1
                        last_score = t
                    if p == prn:
                        rank = current_rank
                        break
            else:
                class_size = None
                avg_total = None
                rank = None

            rows_out.append(
                {
                    "subject_id": subject_id,
                    "subject_name": subject_name,
                    "theory_marks": theory_marks,           # None => Absent
                    "lab_marks": lab_marks,                 # None => Not uploaded
                    "total": total,
                    "status": status,
                    "class_avg_total": avg_total,
                    "rank_total": rank,
                    "class_size": class_size,
                }
            )

        # summary (overall)
        totals_present = [r["total"] for r in rows_out if r.get("total") is not None]
        overall_total = round(sum(totals_present), 2) if totals_present else 0.0
        overall_max = (theory_max + lab_max) * len(rows_out)
        overall_percent = round((overall_total / overall_max) * 100.0, 2) if overall_max > 0 else 0.0

        failed_subjects = [r["subject_name"] for r in rows_out if r.get("status") == "FAIL"]
        low_subjects = [r["subject_name"] for r in rows_out if (r.get("total") is not None and r["total"] < 20)]

        return {
            "rows": rows_out,
            "summary": {
                "subjects_count": len(rows_out),
                "overall_total": overall_total,
                "overall_max": overall_max,
                "overall_percent": overall_percent,
                "failed_subjects": failed_subjects,
                "below_20_subjects": low_subjects,
            },
        }

    finally:
        conn.close()
