"""Generate a CLM test report as .docx, .html, and .pdf.

Based on ``generate clm test report wpython-docx.ipynb``.

Pass/fail limits are ``spec_config_dict``. Each entry is

    metric name -> [[low, high], unit]

A channel passes when ``low <= result <= high``.

Needs ``python-docx``, ``mammoth``, and ``pywin32`` (that last one provides
``win32com``; there is no package named ``win32com``):

    pip install python-docx mammoth pywin32
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# [low, high] is inclusive. Unit is printed in the section heading.
spec_config_dict = {
    "wavelength error": [[-0.1, 0.1], "nm"],
    "power error": [[-0.2, 0.2], "dB"],
    "15 min power stability": [[0, 0.1], "dB"],
    "min PER": [[15, 99], "dB"],
}

TILES = [3, 4]
BANKS = ["A", "B"]
CHANNELS = range(8)
RESULT_COLUMNS = ["tile", "bank", "channel", "spec_lo", "spec_hi", "result", "pass/fail"]

PASS_FILL = "C6EFCE"
FAIL_FILL = "FFC7CE"
HEADER_FILL = "D9E2F3"
PASS_FONT = RGBColor(0x00, 0x61, 0x00)
FAIL_FONT = RGBColor(0x9C, 0x00, 0x06)


def judge(value, spec_lo, spec_hi) -> str:
    """Return 'P' when value is inside the inclusive spec window."""
    if value is None or pd.isna(value):
        return "F"
    return "P" if spec_lo <= float(value) <= spec_hi else "F"


def _round_result(value, ndigits: int = 3):
    """Round numeric report results to ``ndigits`` decimal places."""
    if value is None or pd.isna(value):
        return value
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def _format_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    # Tile / channel (and other whole numbers) as integers, not 3.000.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
        return f"{as_float:.3f}"
    return str(value)


def _coerce_tile_channel_ints(results):
    """Ensure ``tile`` and ``channel`` columns are integers when present."""
    if not isinstance(results, pd.DataFrame) or results.empty:
        return results
    frame = results.copy()
    for col in ("tile", "channel"):
        if col not in frame.columns:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    return frame


def _lookup_result(results, metric, tile, bank, channel):
    """Read one measurement.

    ``results`` may be omitted (every result is 0, matching the notebook
    template), a DataFrame with one column per metric, or a dict of
    ``(tile, bank, channel) -> value`` maps keyed by metric name.
    """
    if results is None:
        return 0

    if isinstance(results, pd.DataFrame):
        frame = results
        if metric not in frame.columns:
            return None
        match = frame[
            (frame["tile"] == tile)
            & (frame["bank"] == bank)
            & (frame["channel"] == channel)
        ]
        if match.empty:
            return None
        return match.iloc[0][metric]

    metric_map = results.get(metric, {})
    if isinstance(metric_map, pd.DataFrame):
        return _lookup_result(metric_map.rename(columns={"result": metric}), metric, tile, bank, channel)
    return metric_map.get((tile, bank, channel))


def _result_slots(results):
    """Tile, bank, and channel rows to print.

    A supplied result fills only the slots it contains. With no result, the
    table stays the full tile 3 and tile 4 grid used by the blank template.
    """
    if results is None:
        return [
            (tile, bank, channel)
            for tile in TILES
            for bank in BANKS
            for channel in CHANNELS
        ]

    if isinstance(results, pd.DataFrame):
        needed = {"tile", "bank", "channel"}
        if results.empty or not needed <= set(results.columns):
            return []
        slots = results.loc[:, ["tile", "bank", "channel"]].drop_duplicates()
        slots = slots.sort_values(["tile", "bank", "channel"])
        return [
            (int(tile), str(bank), int(channel))
            for tile, bank, channel in slots.itertuples(index=False, name=None)
        ]

    found = set()
    for metric_map in results.values():
        if isinstance(metric_map, pd.DataFrame):
            found.update(_result_slots(metric_map))
        else:
            found.update(metric_map.keys())
    return sorted(found, key=lambda item: (item[0], str(item[1]), item[2]))


def build_result_table(metric, spec, results=None) -> pd.DataFrame:
    spec_lo, spec_hi = spec[0]
    rows = []
    results = _coerce_tile_channel_ints(results) if isinstance(results, pd.DataFrame) else results
    for tile, bank, channel in _result_slots(results):
        value = _round_result(_lookup_result(results, metric, tile, bank, channel))
        rows.append(
            {
                "tile": int(tile),
                "bank": bank,
                "channel": int(channel),
                "spec_lo": spec_lo,
                "spec_hi": spec_hi,
                "result": value,
                "pass/fail": judge(value, spec_lo, spec_hi),
            }
        )
    out = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if len(out):
        out["tile"] = out["tile"].astype(int)
        out["channel"] = out["channel"].astype(int)
    return out


def build_all_results(spec_config, results=None) -> dict[str, pd.DataFrame]:
    if isinstance(results, pd.DataFrame):
        results = _coerce_tile_channel_ints(results)
    return {
        metric: build_result_table(metric, spec, results)
        for metric, spec in spec_config.items()
    }


def _shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def _set_cell(cell, text, *, bold=False, fill=None, font_color=None, align="center") -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER if align == "center" else WD_ALIGN_PARAGRAPH.LEFT
    )
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = Pt(10)
    run.font.name = "Calibri"
    if font_color is not None:
        run.font.color.rgb = font_color
    if fill:
        _shade_cell(cell, fill)


def _mark_header_row(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:tblHeader"))


def _add_key_value_table(doc, pairs) -> None:
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    table.autofit = False
    for key, value in pairs:
        cells = table.add_row().cells
        _set_cell(cells[0], str(key), bold=True, align="left", fill=HEADER_FILL)
        _set_cell(cells[1], str(value), align="left")
        cells[0].width = Inches(2.2)
        cells[1].width = Inches(4.3)


def _add_result_table(doc, frame: pd.DataFrame) -> None:
    table = doc.add_table(rows=0, cols=len(RESULT_COLUMNS))
    table.style = "Table Grid"
    table.autofit = False

    header = table.add_row()
    _mark_header_row(header)
    for cell, column in zip(header.cells, RESULT_COLUMNS):
        _set_cell(cell, column, bold=True, fill=HEADER_FILL)

    for record in frame.itertuples(index=False):
        cells = table.add_row().cells
        verdict = record[-1]
        values = [
            record.tile,
            record.bank,
            record.channel,
            record.spec_lo,
            record.spec_hi,
            record.result,
            verdict,
        ]
        for index, value in enumerate(values):
            fill = None
            font_color = None
            bold = False
            if index == len(values) - 1:
                bold = True
                if verdict == "P":
                    fill, font_color = PASS_FILL, PASS_FONT
                else:
                    fill, font_color = FAIL_FILL, FAIL_FONT
            _set_cell(
                cells[index],
                _format_value(value),
                bold=bold,
                fill=fill,
                font_color=font_color,
            )


def _summary_rows(tables: dict[str, pd.DataFrame], spec_config) -> list[tuple]:
    rows = []
    for metric, frame in tables.items():
        unit = spec_config[metric][1]
        fails = int((frame["pass/fail"] == "F").sum())
        status = "P" if fails == 0 else "F"
        rows.append((metric, unit, len(frame), fails, status))
    return rows


def build_docx(dict_test, dut, tables, spec_config) -> Document:
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)

    title = doc.add_heading("CLM Test Report", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    summary = _summary_rows(tables, spec_config)
    total_fails = sum(item[3] for item in summary)
    overall = "PASS" if total_fails == 0 else "FAIL"
    verdict = doc.add_paragraph()
    verdict.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = verdict.add_run(f"Overall: {overall}")
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = PASS_FONT if overall == "PASS" else FAIL_FONT

    doc.add_heading("Meta Data", level=2)
    _add_key_value_table(doc, list(dict_test.items()) + list(dut.items()))

    doc.add_heading("summary", level=2)
    summary_table = doc.add_table(rows=1, cols=5)
    summary_table.style = "Table Grid"
    for cell, column in zip(
        summary_table.rows[0].cells,
        ["metric", "unit", "total count", "fail count", "pass/fail"],
    ):
        _set_cell(cell, column, bold=True, fill=HEADER_FILL)
    _mark_header_row(summary_table.rows[0])
    for metric, unit, count, fails, status in summary:
        cells = summary_table.add_row().cells
        fill = PASS_FILL if status == "P" else FAIL_FILL
        font_color = PASS_FONT if status == "P" else FAIL_FONT
        for cell, text in zip(cells, [metric, unit, count, fails, status]):
            _set_cell(
                cell,
                _format_value(text),
                bold=(text == status),
                fill=fill if text == status else None,
                font_color=font_color if text == status else None,
                align="left" if text == metric else "center",
            )

    for metric, frame in tables.items():
        unit = spec_config[metric][1]
        doc.add_heading(f"{metric} ({unit}):", level=2)
        _add_result_table(doc, frame)

    return doc


def _color_verdict_cells(fragment: str) -> str:
    for verdict, css in (("P", "pass"), ("F", "fail")):
        for inner in (f"<p><strong>{verdict}</strong></p>", f"<p>{verdict}</p>"):
            fragment = fragment.replace(
                f"<td>{inner}</td>",
                f'<td class="{css}">{inner}</td>',
            )
            fragment = fragment.replace(
                f"<th>{inner}</th>",
                f'<th class="{css}">{inner}</th>',
            )
    return fragment


def write_html(docx_path: Path, html_path: Path) -> None:
    import mammoth

    with docx_path.open("rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
    if result.messages:
        print("HTML conversion warnings:", result.messages)

    body = _color_verdict_cells(result.value)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(docx_path.stem)}</title>
<style>
body {{ font-family: Calibri, Arial, sans-serif; margin: 24px; color: #222; }}
h1 {{ text-align: center; }}
table {{ border-collapse: collapse; margin: 8px 0 20px; }}
td, th {{ border: 1px solid #666; padding: 4px 8px; }}
td.pass {{ background: #{PASS_FILL}; color: #006100; font-weight: bold; }}
td.fail {{ background: #{FAIL_FILL}; color: #9C0006; font-weight: bold; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")


def write_pdf(docx_path: Path, pdf_path: Path) -> bool:
    """Export the Word report to PDF through the installed Word application.

    Returns False when Word is not installed. The docx and html files are
    still written.
    """
    try:
        import win32com.client
        import pywintypes
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PDF export needs pywin32. Install it with: pip install pywin32"
        ) from exc
    try:
        word = win32com.client.DispatchEx("Word.Application")
    except pywintypes.com_error as exc:
        print(f"PDF skipped. Microsoft Word is not installed ({exc.args[1]}).")
        return False
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(docx_path), ReadOnly=True)
        document.SaveAs(str(pdf_path), FileFormat=17)  # wdFormatPDF
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()
    return True


def report_stem(serial_number: str, test_date: str) -> str:
    safe_date = str(test_date).replace("/", "-")
    return f"clm test report_{serial_number}_{safe_date}"


def generate_clm_test_report(
    dict_test,
    dut,
    spec_config=None,
    results=None,
    output_dir=".",
    file_suffix="",
) -> dict[str, Path]:
    """Write the CLM report and return the docx, html, and pdf paths.

    ``file_suffix`` is appended to the file stem, before the extension.
    """
    spec_config = spec_config_dict if spec_config is None else spec_config
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = build_all_results(spec_config, results)
    doc = build_docx(dict_test, dut, tables, spec_config)

    stem = report_stem(dut["Serial Number"], dict_test["test date"]) + str(file_suffix or "")
    docx_path = (output_dir / f"{stem}.docx").resolve()
    try:
        doc.save(docx_path)
    except PermissionError:
        stem = f"{stem}_generated"
        docx_path = (output_dir / f"{stem}.docx").resolve()
        doc.save(docx_path)
        print(f"Target docx is open; wrote {docx_path.name} instead.")
    html_path = (output_dir / f"{stem}.html").resolve()
    pdf_path = (output_dir / f"{stem}.pdf").resolve()
    write_html(docx_path, html_path)
    pdf_written = write_pdf(docx_path, pdf_path)

    for metric, frame in tables.items():
        fails = int((frame["pass/fail"] == "F").sum())
        print(f"{metric}: {len(frame) - fails} P, {fails} F")
    print(docx_path)
    print(html_path)
    if pdf_written:
        print(pdf_path)
    return {"docx": docx_path, "html": html_path, "pdf": pdf_path if pdf_written else None}


if __name__ == "__main__":
    dict_test = {
        "test date": "10/19/2026",
        "tester": "xing pan",
        "target power": "12 mW/ch",
    }
    serial_number = "Y2611000977"
    dut = {"Serial Number": serial_number, "Part Number": "nnnn"}
    generate_clm_test_report(
        dict_test,
        dut,
        spec_config_dict,
        output_dir=Path(__file__).resolve().parent,
    )
