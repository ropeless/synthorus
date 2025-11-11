def unindent(s: str) -> str:
    """
    Removing indentation from triple quoted strings.
    """
    lines = s.splitlines()
    if lines[0] == '':
        del lines[0]
    prefix = min(len(l) - len(l.lstrip()) for l in lines)
    return '\n'.join(l[prefix:] for l in lines)


def strip_lines(s: str) -> str:
    """
    Strip all lines, and delete empty lines.
    """
    lines = filter(
        lambda l: l != '',
        (line.strip() for line in s.splitlines())
    )
    return '\n'.join(lines)
