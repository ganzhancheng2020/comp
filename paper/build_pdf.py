"""Render paper/paper.md to paper/paper.pdf with headless Chromium."""
import os
import re
import subprocess

import markdown

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "paper.md")).read()
m = re.match(r"---\ntitle: \"(.*?)\"\nsubtitle: \"(.*?)\"\n---\n", src)
title, subtitle = m.group(1), m.group(2)
body = markdown.markdown(src[m.end():], extensions=["tables"])
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
@page {{ size: Letter; margin: 0.8in; }}
body {{ font-family: 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.38; color: #111; }}
h1 {{ font-size: 17pt; text-align: center; margin: 0 0 4px; }}
.sub {{ text-align: center; color: #444; margin-bottom: 18px; }}
h2 {{ font-size: 12.5pt; margin: 16px 0 6px; }} h3 {{ font-size: 11pt; margin: 12px 0 4px; }}
table {{ border-collapse: collapse; margin: 8px auto; font-size: 9.5pt; }}
th, td {{ border-top: 1px solid #999; border-bottom: 1px solid #999; padding: 3px 8px; text-align: left; }}
th {{ border-top: 1.5px solid #000; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; }}
</style></head><body><h1>{title}</h1><div class="sub">{subtitle}</div>{body}</body></html>"""
out_html = os.path.join(here, "paper.html")
open(out_html, "w").write(html)
chrome = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", f"--print-to-pdf={os.path.join(here, 'paper.pdf')}",
                "--no-pdf-header-footer", out_html], check=True, capture_output=True)
print("wrote paper.pdf")
