"""Convert the release's Markdown files to plain text and remove the .md copies (backed up next to this script)."""
from pathlib import Path
import re
import shutil
import textwrap

PACKAGE = Path('/path/to/GECC/uploadGECC/census_tracts_12_metrics_20261102')
HERE = Path(__file__).resolve().parent
NAMES = ('README', 'FILE_GUIDE', 'RELEASE_NOTES')
WIDTH = 100


def rename(s):
    for n in NAMES:
        s = s.replace(f'{n}.md', f'{n}.txt')
    return s.replace('three Markdown documents', 'three text documents')


def inline(s):
    def link(m):
        text, target = m.group(1), m.group(2)
        if text == target:
            return text
        if target.endswith('.zip') and not text.endswith('.zip'):
            return f'{text} (in {target})'
        return f'{text} ({target})'
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, s)
    s = s.replace('**', '').replace('`', '')
    return s


def wrap(s, first='', rest=''):
    return textwrap.fill(s, WIDTH, initial_indent=first, subsequent_indent=rest, break_on_hyphens=False, break_long_words=False)


def table(lines):
    rows = [[c.strip() for c in l.strip().strip('|').split('|')] for l in lines]
    head, body = rows[0], [r for r in rows[2:]]
    out = []
    for r in body:
        out.append(inline(r[0]))
        if len(r) == 2:
            out.append(wrap(inline(r[1]), '    ', '    '))
        else:
            for h, c in zip(head[1:], r[1:]):
                out.append(wrap(f'{h}: {inline(c)}', '    ', '      '))
        out.append('')
    return out


def convert(md):
    lines, out, i = md.split('\n'), [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            i += 1
            while not lines[i].startswith('```'):
                out.append('    ' + lines[i]); i += 1
            i += 1; continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i]); i += 1
            out += table(block); continue
        m = re.match(r'(#+) (.*)', line)
        if m:
            text = inline(m.group(2))
            out += [text, {1: '=', 2: '-'}.get(len(m.group(1)), '~') * len(text)]
        elif line.startswith('- '):
            out.append(wrap(inline(line[2:]), '- ', '  '))
        elif line.strip():
            out.append(wrap(inline(line)))
        else:
            out.append('')
        i += 1
    text = '\n'.join(out)
    return re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'


backup = HERE / 'markdown_before_txt'
backup.mkdir(exist_ok=True)
for n in NAMES:
    src = PACKAGE / f'{n}.md'
    shutil.copy2(src, backup / src.name)
    (PACKAGE / f'{n}.txt').write_text(convert(rename(src.read_text(encoding='utf-8'))), encoding='utf-8')
    src.unlink()
    print('converted', n)
