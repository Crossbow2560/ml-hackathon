"""Build report/PathSnap_Report.pdf from report/report.md.

1. Fill placeholders from the result files (report/make_tables.py):
       {{table:<name>}}  {{v:<model>:<column>[:fmt]}}  {{j:<json file>:<key.path>[:fmt]}}
2. Markdown -> HTML (tables, code blocks), with print CSS for A4.
3. HTML -> PDF with headless Chromium (no extra Python dependency).
Run: python -m report.build_report
"""
from __future__ import annotations

import re
import shutil
import subprocess

import markdown

from pathsnap.io import REPO_ROOT
from report.make_tables import TABLES, json_value, value

DIR = REPO_ROOT / "report"
CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #111; }
h1 { font-size: 20pt; margin-bottom: 2pt; } h2 { font-size: 14pt; border-bottom: 1px solid #999; margin-top: 18pt; }
h3 { font-size: 12pt; margin-top: 12pt; } h4 { font-size: 10.5pt; margin: 10pt 0 2pt; }
table { border-collapse: collapse; font-size: 7.6pt; margin: 6pt 0; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { background: #eef2f7; }
img { max-width: 100%; max-height: 120mm; display: block; margin: 6pt auto 2pt; page-break-inside: avoid; }
p > em:only-child { font-size: 8.5pt; color: #444; }
pre { background: #f4f4f4; padding: 6pt; font-size: 7.5pt; white-space: pre-wrap; }
code { font-size: 8.5pt; }
"""


def fill(text: str) -> str:
    text = re.sub(r"\{\{table:(\w+)\}\}", lambda m: "\n" + TABLES[m.group(1)]() + "\n", text)
    text = re.sub(r"\{\{v:(\w+):(\w+)(?::([^}]*))?\}\}",
                  lambda m: value(m.group(1), m.group(2), m.group(3) or ".4f"), text)
    text = re.sub(r"\{\{j:([^:}]+):([^:}]+)(?::([^}]*))?\}\}",
                  lambda m: json_value(m.group(1), m.group(2), m.group(3) or ".4f"), text)
    left = re.findall(r"\{\{[^}]*\}\}", text)
    assert not left, f"unfilled placeholders: {left}"
    return text


def main():
    md_text = fill((DIR / "report.md").read_text())
    (DIR / "report_filled.md").write_text(md_text)
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap report</title><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = DIR / "PathSnap_Report.html"
    html_path.write_text(html)
    browser = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if not browser:
        raise SystemExit(f"HTML written to {html_path}; install Chromium/Chrome to print the PDF")
    pdf = DIR / "PathSnap_Report.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, capture_output=True)
    print(f"wrote {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
