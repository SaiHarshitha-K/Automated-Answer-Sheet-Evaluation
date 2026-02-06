# subject_report.py
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st
from io import BytesIO


# -----------------------------
# Helpers
# -----------------------------
def _collect_theory_marks(subject_result: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    for prn, attempts in subject_result["students"].items():
        # take BEST attempt
        best = max(attempts, key=lambda x: x["score"])
        rows.append({
            "PRN": prn,
            "Name": best["name"],
            "Theory": best["score"],
        })

    return pd.DataFrame(rows)


#def _load_lab_marks(lab_excel) -> Optional[pd.DataFrame]:
#    if lab_excel is None:
#        return None
#
#    df = pd.read_excel(lab_excel)
#    df.columns = ["PRN", "Lab"]
#    df["PRN"] = df["PRN"].astype(str).str.strip()
#    return df




def _question_wrong_stats(subject_result: Dict[str, Any], key_map: Dict[int, str]):
    wrong_count = {q: 0 for q in key_map}

    for attempts in subject_result["students"].values():
        for a in attempts:
            for d in a["details"]:
                if not d["is_correct"]:
                    wrong_count[d["question"]] += 1

    return sorted(wrong_count.items(), key=lambda x: x[1], reverse=True)


# -----------------------------
# MAIN API
# -----------------------------
def render_subject_report(
    *,
    subject_name: str,
    subject_result: Dict[str, Any],
    key_map: Dict[int, str],
    lab_marks_map=None,
):
    st.subheader("📊 Subject Report")

    theory_df = _collect_theory_marks(subject_result)

    # ✅ Build lab_df from DB map: {PRN: marks}
    # ✅ Build lab_df from DB map: {PRN: marks}
    lab_df = None
    lab_applicable = False  # ✅ only true if at least one real lab mark exists

    if lab_marks_map is not None:
        lab_rows = []
        for prn, marks in lab_marks_map.items():
            prn_norm = str(prn).strip().upper()
            # marks can be None (NULL) or -1 (ABSENT) or number
            lab_rows.append({"PRN": prn_norm, "Lab": marks})
            if marks is not None and float(marks) != -1:
                lab_applicable = True  # ✅ lab uploaded (real marks exist)

        lab_df = pd.DataFrame(lab_rows) if lab_rows else None

    # Merge theory + lab (keep NaN for missing students)
    # -----------------------------
    # LAB (from DB)
    # -----------------------------
    df = theory_df.copy()
    df["PRN"] = df["PRN"].astype(str).str.strip()

    lab_applicable = False
    lab_df = None

    if lab_marks_map is not None:
        lab_rows = []
        for prn, marks in lab_marks_map.items():
            prn_norm = str(prn).strip()
            lab_rows.append({"PRN": prn_norm, "Lab": marks})

            # ✅ If at least one real mark exists => lab is applicable
            if marks is not None:
                lab_applicable = True

        if lab_rows:
            lab_df = pd.DataFrame(lab_rows)

    if lab_df is not None and not lab_df.empty:
        df = df.merge(lab_df, on="PRN", how="left")
    else:
        df["Lab"] = None

    # ✅ display AB if NaN/NULL (when lab is applicable)
    def _lab_display(x):
        if pd.isna(x):
            return "AB" if lab_applicable else 0
        return float(x)

    df["Lab_Display"] = df["Lab"].apply(_lab_display)

    # ✅ Total: AB counts as 0 only for total calculation
    def _lab_for_total(x):
        return 0 if x == "AB" else float(x)

    df["Total"] = df["Theory"] + df["Lab_Display"].apply(_lab_for_total)

    # ✅ Status logic
    def _status(row):
        if row["Theory"] < 16:
            return "FAIL"

        if not lab_applicable:
            return "PASS"   # lab not conducted

        # lab conducted => NaN becomes AB => ABSENT
        if row["Lab_Display"] == "AB":
            return "ABSENT"

        if float(row["Lab_Display"]) < 16:
            return "FAIL"

        return "PASS"

    df["Status"] = df.apply(_status, axis=1)

    df = df.drop(columns=["Lab"]).rename(columns={"Lab_Display": "Lab"})

    # -----------------------------
    # SUMMARY POPUP
    # -----------------------------
    with st.expander("📌 Summary Report", expanded=True):

        st.markdown("### 🏆 Top 5 – Theory")
        st.dataframe(df.sort_values("Theory", ascending=False).head(5))

        if lab_df is not None:
            st.markdown("### 🧪 Top 5 – Lab")
            st.dataframe(df.sort_values("Lab", ascending=False).head(5))

            st.markdown("### 🧮 Top 5 – Total")
            st.dataframe(df.sort_values("Total", ascending=False).head(5))

        failed = df[df["Status"] == "FAIL"]
        if not failed.empty:
            st.markdown("### ❌ Failed Students (PRN)")
            st.write(", ".join(failed["PRN"].tolist()))

        st.markdown("### ❓ Most Wrong Questions")
        wrong_qs = _question_wrong_stats(subject_result, key_map)[:5]
        for q, c in wrong_qs:
            st.write(f"Q{q} → {c} students wrong")

    # -----------------------------
    # EXCEL EXPORT
    # -----------------------------
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Report")

        workbook = writer.book
        worksheet = writer.sheets["Report"]

        red = workbook.add_format({"bg_color": "#FFC7CE"})

        for i, status in enumerate(df["Status"], start=1):
            if status == "FAIL":
                worksheet.set_row(i, None, red)

    st.download_button(
        "⬇ Download Subject Report (Excel)",
        data=buffer.getvalue(),
        file_name=f"{subject_name}_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
