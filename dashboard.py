# dashboard.py
import streamlit as st

from db_utils import (
    get_connection,
    get_batches,
    get_courses,
)

# ------------------------------------------------------------
# Small DB helpers (fast counts + subject status lists)
# ------------------------------------------------------------
def _fetch_course_overview(course_id: int) -> dict:
    """
    Returns:
      {
        students_total: int,
        subjects_total: int,
        subjects_with_exam: int,
        published_subjects: int
      }
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # students
        cur.execute("SELECT COUNT(*) FROM students WHERE course_id=%s", (course_id,))
        (students_total,) = cur.fetchone()

        # subjects
        cur.execute("SELECT COUNT(*) FROM subjects WHERE course_id=%s", (course_id,))
        (subjects_total,) = cur.fetchone()

        # subjects WITH at least 1 exam (latest doesn't matter for count)
        cur.execute(
            """
            SELECT COUNT(DISTINCT s.id)
            FROM subjects s
            JOIN exams e ON e.subject_id = s.id
            WHERE s.course_id = %s
            """,
            (course_id,),
        )
        (subjects_with_exam,) = cur.fetchone()

        # published subjects
        cur.execute(
            """
            SELECT COUNT(*)
            FROM subject_publish sp
            JOIN subjects s ON s.id = sp.subject_id
            WHERE s.course_id=%s AND sp.is_published=1
            """,
            (course_id,),
        )
        (published_subjects,) = cur.fetchone()

        return {
            "students_total": int(students_total or 0),
            "subjects_total": int(subjects_total or 0),
            "subjects_with_exam": int(subjects_with_exam or 0),
            "published_subjects": int(published_subjects or 0),
        }
    finally:
        conn.close()


def _fetch_subject_status_rows(course_id: int) -> list[dict]:
    """
    List subjects in a course with:
      - has_exam (any exam exists)
      - is_published (subject_publish.is_published=1)
      - latest_exam_id (optional)
      - published_exam_id (optional)
    """
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT
                s.id AS subject_id,
                s.name AS subject_name,

                -- does any exam exist?
                CASE WHEN EXISTS (
                    SELECT 1 FROM exams e WHERE e.subject_id = s.id LIMIT 1
                ) THEN 1 ELSE 0 END AS has_exam,

                -- latest exam id (if exists)
                (
                    SELECT e2.id
                    FROM exams e2
                    WHERE e2.subject_id = s.id
                    ORDER BY e2.created_at DESC
                    LIMIT 1
                ) AS latest_exam_id,

                -- publish flag
                COALESCE(sp.is_published, 0) AS is_published,

                -- published exam id (if published)
                CASE WHEN COALESCE(sp.is_published, 0)=1 THEN sp.exam_id ELSE NULL END AS published_exam_id

            FROM subjects s
            LEFT JOIN subject_publish sp
              ON sp.subject_id = s.id
            WHERE s.course_id = %s
            ORDER BY s.id
            """,
            (course_id,),
        )
        rows = cur.fetchall() or []
        # normalize ints/bools
        out = []
        for r in rows:
            out.append(
                {
                    "subject_id": int(r["subject_id"]),
                    "subject_name": r["subject_name"],
                    "has_exam": bool(r["has_exam"]),
                    "latest_exam_id": int(r["latest_exam_id"]) if r["latest_exam_id"] is not None else None,
                    "is_published": bool(r["is_published"]),
                    "published_exam_id": int(r["published_exam_id"]) if r["published_exam_id"] is not None else None,
                }
            )
        return out
    finally:
        conn.close()


# ------------------------------------------------------------
# Main Dashboard UI
# ------------------------------------------------------------
def render_dashboard():
    st.title("📊 Admin Dashboard")

    batches = get_batches()
    if not batches:
        st.info("No batches found. Create a batch first in Batches / Courses / Subjects.")
        return

    batch_options = {name: bid for bid, name in batches}
    selected_batch_name = st.selectbox(
        "Select Batch",
        ["-- choose --"] + list(batch_options.keys()),
        key="dash_batch_select",
    )
    batch_id = batch_options.get(selected_batch_name)

    if not batch_id:
        st.info("Select a batch to view courses summary.")
        return

    st.markdown("---")
    st.subheader(f"📚 Courses in {selected_batch_name}")

    courses = get_courses(batch_id)
    if not courses:
        st.warning("No courses in this batch yet.")
        return

    # Show each course as an expander with metrics + subjects + course report button
    for course_id, course_name in courses:
        course_id = int(course_id)

        with st.expander(f"🎓 {course_name}", expanded=False):
            overview = _fetch_course_overview(course_id)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Students Added", overview["students_total"])
            c2.metric("Subjects", overview["subjects_total"])
            c3.metric("Subjects with Exam", overview["subjects_with_exam"])
            c4.metric("Published Subjects", overview["published_subjects"])

            st.markdown("### 📌 Subjects Status")
            subject_rows = _fetch_subject_status_rows(course_id)

            if not subject_rows:
                st.info("No subjects yet for this course.")
            else:
                # lightweight table-like rendering (faster than dataframe for many rows)
                for r in subject_rows:
                    left, mid, right = st.columns([2.2, 1.2, 1.3])

                    exam_badge = "✅ Exam" if r["has_exam"] else "⏳ No Exam"
                    pub_badge = "✅ Published" if r["is_published"] else "⏳ Not posted"

                    left.write(f"**{r['subject_name']}** (ID: {r['subject_id']})")
                    mid.write(exam_badge)
                    right.write(pub_badge)

                    # optional tiny details
                    if r["has_exam"]:
                        st.caption(
                            f"Latest Exam ID: {r['latest_exam_id']} | "
                            f"Published Exam ID: {r['published_exam_id'] if r['published_exam_id'] else '-'}"
                        )

            st.markdown("---")
            st.markdown("### 📄 Course Report (existing)")
            pass_percent = st.number_input(
                "Pass % threshold",
                min_value=0.0,
                max_value=100.0,
                value=35.0,
                step=1.0,
                key=f"dash_pass_{course_id}",
            )

            if st.button("📊 Generate Course Report", key=f"dash_course_report_{course_id}"):
                # import here to avoid circular imports
                from course_report import render_course_report
                render_course_report(course_id=course_id, pass_percent=pass_percent)
