import os
import re
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch


def parse_markdown(md_text):
    lines = md_text.splitlines()
    i = 0
    blocks = []
    table_mode = False
    table_buf = []

    img_regex = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            if table_mode:
                blocks.append(('table', table_buf))
                table_buf = []
                table_mode = False
            i += 1
            continue

        # table detection (lines that start and contain pipes)
        if line.strip().startswith('|'):
            # collect contiguous pipe-lines
            table_mode = True
            table_buf.append(line)
            i += 1
            # if next lines continue, loop will append
            continue

        # image
        m = img_regex.search(line)
        if m:
            alt, path = m.groups()
            blocks.append(('image', path.strip()))
            i += 1
            continue

        # heading
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            blocks.append(('heading', level, text))
            i += 1
            continue

        # paragraph - gather until blank or special
        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('|') and not img_regex.search(lines[i]) and not lines[i].lstrip().startswith('#'):
            para_lines.append(lines[i].rstrip())
            i += 1
        blocks.append(('para', ' '.join(para_lines).strip()))

    # flush table if file ends with table
    if table_mode and table_buf:
        blocks.append(('table', table_buf))

    return blocks


def table_from_md(table_lines):
    rows = []
    for ln in table_lines:
        # strip leading/trailing pipe
        parts = [p.strip() for p in ln.strip().strip('|').split('|')]
        rows.append(parts)
    # If separator row (---) present, remove it
    if len(rows) >= 2 and all(re.match(r'^:?-{3,}:?$', c) for c in rows[1]):
        rows.pop(1)
    return rows


def build_pdf(md_path, out_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    blocks = parse_markdown(text)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # custom styles
    h1 = ParagraphStyle('Heading1', parent=styles['Heading1'], spaceAfter=12)
    h2 = ParagraphStyle('Heading2', parent=styles['Heading2'], spaceAfter=8)
    normal = styles['BodyText']
    normal.spaceAfter = 6

    md_dir = os.path.dirname(md_path)
    max_image_width = doc.width
    page_w, page_h = doc.pagesize
    max_image_height = page_h - doc.topMargin - doc.bottomMargin - 40

    for block in blocks:
        if block[0] == 'heading':
            _, level, text = block
            if level == 1:
                story.append(Paragraph(text, h1))
            else:
                story.append(Paragraph(text, h2))
        elif block[0] == 'para':
            _, text = block
            # simple replacement for inline code markers
            text = text.replace('`', '')
            story.append(Paragraph(text, normal))
        elif block[0] == 'image':
            _, path = block
            img_path = os.path.join(md_dir, path)
            if not os.path.exists(img_path):
                # try relative to repo root
                img_path = os.path.abspath(path)
            if not os.path.exists(img_path):
                story.append(Paragraph(f"[Missing image: {path}]", normal))
            else:
                try:
                    img = Image(img_path)
                    # scale image to fit width
                    iw, ih = img.wrap(0, 0)
                    scale_w = 1.0
                    scale_h = 1.0
                    if iw > max_image_width:
                        scale_w = max_image_width / float(iw)
                    if ih > max_image_height:
                        scale_h = max_image_height / float(ih)
                    scale = min(scale_w, scale_h)
                    if scale < 1.0:
                        img.drawWidth = iw * scale
                        img.drawHeight = ih * scale
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    story.append(Paragraph(f"[Could not embed image: {path} - {e}]", normal))
        elif block[0] == 'table':
            _, table_lines = block
            rows = table_from_md(table_lines)
            tbl = Table(rows, hAlign='LEFT')
            tbl.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 12))

    doc.build(story)


def main():
    if len(sys.argv) < 2:
        print('Usage: python md_to_pdf.py path/to/file.md [out.pdf]')
        sys.exit(1)
    md_path = sys.argv[1]
    if not os.path.exists(md_path):
        print('Markdown file not found:', md_path)
        sys.exit(1)
    out_path = sys.argv[2] if len(sys.argv) >= 3 else os.path.splitext(md_path)[0] + '.pdf'
    build_pdf(md_path, out_path)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
