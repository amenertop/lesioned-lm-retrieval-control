"""Split cairo's batched text objects in a PDF figure so every text string is its own BT/ET object.

R's cairo_pdf device (cairo in general) keeps one text object open across consecutive text calls: all tick labels
of an axis, or the strip labels of neighbouring facets, land in one BT ... ET with Tm, Td and Tf changes between
them, and a string in a second font (the minus sign) continues from the advanced text position without any move.
PyMuPDF's get_texttrace() reports one trace per text object and font run, and the collision audit of the
nature-figure skill unions each label with its trace, so every tick label of an axis is reported as one box that
overlaps all the others (false text-text failures). This rewrites the page's content stream so that each Tj or TJ
stands in its own text object, placed with the absolute origin of its first glyph as the text trace measured it
(the font size, skew and rotation of the current text matrix are kept; whether the page space is y down, as cairo
writes it after a global flip, or y up is read off the first string and asserted at every string that follows its
own Tm). The rendering is unchanged: --check
renders the page before and after at 300 dpi and refuses the rewrite if more than a handful of pixels differ.

Usage: python split_pdf_text.py figure.pdf [--check]
"""
import os
import re
import sys
import tempfile

import fitz  # PyMuPDF

OPS = r"Tm|Tf|Tj|TJ|Td|TD|T\*|TL|Tc|Tw|Tz|Ts|Tr|rg|RG|g|G|k|K|cs|CS|sc|scn|SC|SCN|gs|w|J|j|M|d|ri|i"
LINE = re.compile(r"^(.*?)\s*(" + OPS + r")$", re.S)
OBJ = re.compile(rb"BT\n(.*?)\nET", re.S)
TEXT_STATE = ("Tc", "Tw", "Tz", "TL", "Ts", "Tr")
STRING = re.compile(r"\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>", re.S)
ESC = {"n": 10, "r": 13, "t": 9, "b": 8, "f": 12, "(": 40, ")": 41, "\\": 92}


def fmt(v):
    return ("%.6f" % v).rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def n_codes(operand, bytes_per_code):
    """Number of character codes shown by a Tj operand or a TJ array."""
    n = 0
    for s in STRING.findall(operand):
        if s.startswith("<"):
            hexdigits = re.sub(r"\s", "", s[1:-1])
            n += (len(hexdigits) + 1) // 2 // bytes_per_code
        else:
            body, i, count = s[1:-1], 0, 0
            while i < len(body):
                ch = body[i]
                if ch == "\\":
                    nxt = body[i + 1] if i + 1 < len(body) else ""
                    if nxt in ESC:
                        i += 2
                    elif nxt.isdigit():
                        j = i + 1
                        while j < len(body) and j < i + 4 and body[j].isdigit():
                            j += 1
                        i = j
                    elif nxt == "\n":
                        i += 2
                        continue
                    else:
                        i += 2
                    count += 1
                else:
                    i += 1
                    count += 1
            n += count // bytes_per_code
    return n


TJ_ITEM = re.compile(r"\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|[-+]?\d*\.?\d+", re.S)


GAP = 250   # thousandths of an em: kerning inside a label stays far below, the gap between two labels far above


def tj_pieces(operand):
    """Cut a TJ array into the labels cairo packed into it. A label begins at the first non-empty string after an
    adjustment larger than GAP (or at the start); each piece is re-emitted from its first non-empty string, because
    the trace origin of that glyph already includes the preceding adjustment ([<>-626<0001>] in cairo's output)."""
    inner = operand.strip()
    assert inner.startswith("[") and inner.endswith("]"), operand
    items = TJ_ITEM.findall(inner[1:-1])
    pieces, current = [], []
    for item in items:
        is_string = item[0] in "(<"
        if is_string and item in ("()", "<>"):
            continue
        if not is_string and abs(float(item)) > GAP and current:
            pieces.append(current)
            current = []
            continue
        if not is_string and not current:
            continue                              # leading adjustment of a piece
        current.append(item)
    if current:
        pieces.append(current)
    return ["[" + "".join(p) + "]" for p in pieces]


def split_object(body, origins, font_bytes, page_height, convention):
    """One cairo text object (between BT and ET) -> self-contained objects, one per shown string.

    origins: iterator over the glyph origins of the page in stream order (PyMuPDF text trace, page coordinates)."""
    out = []
    tm = None
    font = None
    state = {}
    fresh_tm = False
    for raw in body.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = LINE.match(line)
        if not m:
            raise ValueError("unparsed line in text object: %r" % line)
        operand, op = m.group(1), m.group(2)
        if op == "Tm":
            tm = [float(v) for v in operand.split()]
            assert len(tm) == 6, line
            fresh_tm = True
        elif op in ("Td", "TD", "T*"):
            fresh_tm = False                       # the trace gives every string's absolute origin
        elif op == "Tf":
            font = operand
        elif op in TEXT_STATE:
            state[op] = operand
        elif op in ("Tj", "TJ"):
            assert tm is not None and font is not None, line
            bpc = font_bytes[font.split()[0].lstrip("/")]
            for piece in (tj_pieces(operand) if op == "TJ" else [operand]):
                n = n_codes(piece, bpc)
                if n == 0:
                    continue
                first = next(origins)
                for _ in range(n - 1):
                    next(origins)
                x, y = first[0], first[1]
                if convention["y"] is None:        # cairo flips the page (y down), other writers keep y up
                    convention["y"] = "down" if abs(tm[5] - y) < 0.05 else "up"
                    assert fresh_tm, "the first string must follow a Tm"
                if convention["y"] == "up":
                    y = page_height - y
                if fresh_tm:                       # the stream and the trace agree on this string's origin
                    assert abs(tm[4] - x) < 0.05 and abs(tm[5] - y) < 0.05, (tm, first, convention)
                fresh_tm = False
                head = " ".join(fmt(v) for v in (tm[0], tm[1], tm[2], tm[3], x, y)) + " Tm\n" + font + " Tf\n"
                head += "".join("%s %s\n" % (state[k], k) for k in TEXT_STATE if k in state)
                out.append("BT\n" + head + piece + op + "\nET")
        else:                                      # graphics state operators keep their place and order
            out.append(line)
    return out


def split_pdf(path, check=False):
    doc = fitz.open(path)
    before = [p.get_pixmap(dpi=300) for p in doc] if check else None
    n_in = n_out = 0
    for page in doc:
        assert page.rotation == 0 and page.rect.x0 == 0 and page.rect.y0 == 0, "plain unrotated page expected"
        font_bytes = {f[4]: (2 if f[2] == "Type0" else 1) for f in page.get_fonts()}
        chars = [c for trace in page.get_texttrace() for c in trace["chars"]]
        origins = iter([c[2] for c in chars])
        convention = {"y": None}
        for xref in page.get_contents():
            data = doc.xref_stream(xref)
            pieces, pos = [], 0
            for m in OBJ.finditer(data):
                pieces.append(data[pos:m.start()])
                parts = split_object(m.group(1).decode("latin-1"), origins, font_bytes, page.rect.height, convention)
                n_in += 1
                n_out += sum(p.startswith("BT") for p in parts)
                pieces.append("\n".join(parts).encode("latin-1"))
                pos = m.end()
            pieces.append(data[pos:])
            doc.update_stream(xref, b"".join(pieces))
        assert next(origins, None) is None, "glyph count of the stream and of the text trace differ"
    tmp = tempfile.mktemp(suffix=".pdf", dir=os.path.dirname(os.path.abspath(path)))
    doc.save(tmp, garbage=1, deflate=True)
    doc.close()
    if check:
        after = [p.get_pixmap(dpi=300) for p in fitz.open(tmp)]
        weak = strong = 0       # the trace origins are float32, so an anti-aliased edge may shift by a level or two
        for a, b in zip(before, after):
            assert (a.width, a.height, a.n) == (b.width, b.height, b.n)
            sa, sb = a.samples, b.samples
            for i in range(0, len(sa), a.n):
                if sa[i:i + a.n] != sb[i:i + a.n]:
                    d = max(abs(sa[j] - sb[j]) for j in range(i, i + a.n))
                    if d > 16:
                        strong += 1
                    else:
                        weak += 1
        total = sum(a.width * a.height for a in before)
        if strong or weak > total * 1e-4:
            os.remove(tmp)
            raise SystemExit("split_pdf_text: %d pixels differ clearly and %d faintly of %d at 300 dpi; "
                             "file left unchanged" % (strong, weak, total))
        note = "%d of %d pixels differ at 300 dpi, none by more than 16 levels" % (weak, total)
    else:
        note = "render not checked"
    os.replace(tmp, path)
    print("split_pdf_text: %d text objects -> %d (one per string), %s" % (n_in, n_out, note))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split_pdf(args[0], check="--check" in sys.argv)
