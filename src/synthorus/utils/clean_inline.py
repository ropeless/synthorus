def clean_inline(string: str) -> str:
    """
    Remove extraneous space from the given multi-line string.
    This is very similar to a Swift multi-line string.

    1) The first line is discarded if it is empty.
    2) The indent is removed, which is inferred as the
       minimum leading space count over all lines.
    3) the last line is discarded if it is empty.

    """
    lines = string.split('\n')

    if len(lines[0].strip()) == 0:
        lines.pop(0)

    if len(lines) > 0:
        if len(lines[-1].strip()) == 0:
            lines.pop(-1)

        indent = min(len(line) - len(line.lstrip()) for line in lines)

        lines = [
            line[indent:].rstrip()
            for line in lines
        ]

    return '\n'.join(lines) + '\n'
