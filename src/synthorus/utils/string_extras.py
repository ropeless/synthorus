def unindent(s: str) -> str:
    """
    Removing indentation from triple quoted strings.

    This removes extraneous space from the given multi-line string.
    This is very similar to a Swift multi-line string.

    1) The first line is discarded if it is empty.
    2) The indent is removed, which is inferred as the
       minimum leading space count over all lines.
    3) the last line is discarded if it is empty.

    """
    lines = s.splitlines()
    if len(lines) > 0 and len(lines[0].strip()) == 0:
        lines.pop(0)

    if len(lines) > 0:
        if len(lines[-1].strip()) == 0:
            lines.pop(-1)

        indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip() != '')

        lines = (
            line[indent:].rstrip()
            for line in lines
        )

    return '\n'.join(lines) + '\n'


def strip_lines(s: str) -> str:
    """
    Strip all lines, and delete empty lines.
    The last line is terminated in a new line (unless the result is empty).
    """
    lines = filter(
        lambda l: l != '',
        (line.strip() for line in s.splitlines())
    )
    lines = '\n'.join(lines)

    # Ensure the last (non-empty) line has a return
    if len(lines) > 0:
        lines += '\n'

    return lines
