"""Render report.md to report.pdf with headless Chromium (same approach as paper/build_pdf.py)."""
import os
import re
import subprocess

import markdown

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "report.md")).read()
m = re.match(r"---\ntitle: \"(.*?)\"\nsubtitle: \"(.*?)\"\n---\n", src)
title, subtitle = m.group(1), m.group(2)
body = markdown.markdown(src[m.end():], extensions=["tables"])
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
@page {{ size: A4; margin: 18mm 17mm; }}
body {{ font-family: 'DejaVu Serif', 'Times New Roman', serif; font-size: 9.8pt; line-height: 1.38; color: #111; }}
h1 {{ font-size: 16pt; text-align: center; margin: 0 0 4px; }}
.sub {{ text-align: center; color: #444; margin-bottom: 14px; font-size: 9.5pt; }}
h2 {{ font-size: 12pt; margin: 14px 0 5px; border-bottom: 1px solid #bbb; padding-bottom: 2px; }}
h3 {{ font-size: 10.5pt; margin: 10px 0 4px; }}
h2, h3 {{ break-after: avoid; page-break-after: avoid; }} table {{ page-break-inside: avoid; }}
table {{ border-collapse: collapse; margin: 7px auto; font-size: 8.8pt; }}
th, td {{ border-top: 1px solid #aaa; border-bottom: 1px solid #aaa; padding: 2.5px 7px; text-align: left; }}
th {{ border-top: 1.5px solid #000; background: #f4f4f2; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.3pt; }}
.figrow {{ display: flex; gap: 12px; margin: 8px 0; page-break-inside: avoid; }}
figure {{ flex: 1; margin: 0; }} figure img {{ width: 100%; }}
figcaption {{ font-size: 8.3pt; color: #333; line-height: 1.3; }}
ul {{ margin: 4px 0 4px 18px; padding: 0; }} li {{ margin: 2px 0; }}
</style></head><body><h1>{title}</h1><div class="sub">{subtitle}</div>{body}</body></html>"""
out_html = os.path.join(here, "report.html")
open(out_html, "w").write(html)
chrome = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", f"--print-to-pdf={os.path.join(here, 'report.pdf')}",
                "--no-pdf-header-footer", out_html], check=True, capture_output=True)
print("wrote report.pdf")
