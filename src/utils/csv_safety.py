"""Keep untrusted CSV text literal in spreadsheet importers; numbers stay numeric."""

import unicodedata


def spreadsheet_text(value):
    """Prefix formula-like text with an apostrophe without changing source records.

    CSV quoting protects separators, not spreadsheet formula interpretation.
    Consumers requiring byte-exact labels must use the original JSON, not infer
    identities from this spreadsheet-oriented representation.
    """
    if not isinstance(value, str) or not value:
        return value
    first = 0
    while first < len(value) and (
        value[first].isspace() or unicodedata.category(value[first]) in {"Cc", "Cf"}
    ):
        first += 1
    if (first < len(value) and value[first] in "=+-@") or value[0] in "\t\r\n":
        return "'" + value
    return value


def spreadsheet_row(row):
    return {key: spreadsheet_text(value) for key, value in row.items()}
