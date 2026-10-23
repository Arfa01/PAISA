"""Fail-closed handling for user-visible diagnostic text."""

from __future__ import annotations

import argparse
import re


_URL_USERINFO = re.compile(
    r"(?i)\b(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@"
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r'''(?ix)
    (?P<prefix>
        (?<![A-Za-z0-9_])
        ["']?
        (?:
            [A-Za-z0-9_.-]*(?:password|passwd|pwd|secret|credential|token)
                [A-Za-z0-9_.-]*
            |
            [A-Za-z0-9_.-]*(?:api[_.-]?key|access[_.-]?key|private[_.-]?key)
                [A-Za-z0-9_.-]*
            |
            authorization|auth
        )
        ["']?\s*(?:=|:(?!//))\s*
    )
    (?P<value>
        "[^"\r\n]*"
        |
        '[^'\r\n]*'
        |
        [^\r\n,;&\#)\]}]+
    )
    '''
)
_AUTHORIZATION = re.compile(
    r"(?i)\b(?P<prefix>authorization\s*[:=]\s*[\"']?(?:bearer|basic)\s+)\S+"
)
_JWT = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
_KNOWN_TOKEN = re.compile(
    r"\b(?:ghp_|github_pat_|sk-(?:proj-)?)[A-Za-z0-9_-]{8,}\b"
)
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
_PRIVATE_KEY_HEADER = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_PATH_PREFIX = r"(?:file:(?://)?[\\/]|[A-Za-z]:[\\/]|\\\\|//|/)"
_DOUBLE_QUOTED_ABSOLUTE_PATH = re.compile(
    rf'(?i)"{_PATH_PREFIX}[^"\r\n]*"'
)
_SINGLE_QUOTED_ABSOLUTE_PATH = re.compile(
    rf"(?i)'{_PATH_PREFIX}[^'\r\n]*'"
)
_PATH_TAIL = r'''[^\r\n,;)\]}<>"']+'''
_FILE_URI = re.compile(rf"(?i)\bfile:(?://)?[\\/]{_PATH_TAIL}")
_WINDOWS_ABSOLUTE = re.compile(
    rf"(?i)(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\){_PATH_TAIL}"
)
_FORWARD_SLASH_UNC = re.compile(
    rf"(?<![:/])//[^/\r\n,;)\]}}<>\"']+/{_PATH_TAIL}"
)
_POSIX_ABSOLUTE = re.compile(
    rf"(?<![A-Za-z0-9/])/(?!/){_PATH_TAIL}"
)
_SAFE_CREDENTIAL_VALUES = {
    "<redacted>",
    "<redacted-credential>",
    "<redacted-credentials>",
    "none",
    "null",
    "redacted",
}


class ArgumentParsingError(ValueError):
    """An argparse usage error that callers can render without raw stderr."""


class RedactingArgumentParser(argparse.ArgumentParser):
    """Raise parse errors so a CLI can render them through its redactor."""

    def error(self, message: str) -> None:
        raise ArgumentParsingError(message)


def _redact_assignment(match: re.Match[str]) -> str:
    raw_value = match.group("value")
    stripped = raw_value.strip()
    quote = ""
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        quote = stripped[0]
        stripped = stripped[1:-1].strip()
    if stripped.lower() in _SAFE_CREDENTIAL_VALUES:
        return match.group(0)
    return f"{match.group('prefix')}{quote}<redacted-credential>{quote}"


def _redact_quoted_path(match: re.Match[str]) -> str:
    quote = match.group(0)[0]
    return f"{quote}<redacted-path>{quote}"


def redact_diagnostic(value: object) -> str:
    """Remove paths and likely credential material without echoing matches."""

    text = str(value)
    text = _URL_USERINFO.sub(
        lambda match: f"{match.group('scheme')}<redacted-credentials>@", text
    )
    text = _CREDENTIAL_ASSIGNMENT.sub(_redact_assignment, text)
    text = _AUTHORIZATION.sub(
        lambda match: f"{match.group('prefix')}<redacted-credential>", text
    )
    text = _JWT.sub("<redacted-credential>", text)
    text = _KNOWN_TOKEN.sub("<redacted-credential>", text)
    text = _AWS_ACCESS_KEY.sub("<redacted-credential>", text)
    text = _PRIVATE_KEY_BLOCK.sub("<redacted-credential>", text)
    text = _PRIVATE_KEY_HEADER.sub("<redacted-credential>", text)
    text = _DOUBLE_QUOTED_ABSOLUTE_PATH.sub(_redact_quoted_path, text)
    text = _SINGLE_QUOTED_ABSOLUTE_PATH.sub(_redact_quoted_path, text)
    text = _FILE_URI.sub("<redacted-path>", text)
    text = _WINDOWS_ABSOLUTE.sub("<redacted-path>", text)
    text = _FORWARD_SLASH_UNC.sub("<redacted-path>", text)
    text = _POSIX_ABSOLUTE.sub("<redacted-path>", text)
    return text


def contains_sensitive_text(value: object) -> bool:
    """Return whether redaction would remove path or credential material."""

    text = str(value)
    return redact_diagnostic(text) != text


__all__ = [
    "ArgumentParsingError",
    "RedactingArgumentParser",
    "contains_sensitive_text",
    "redact_diagnostic",
]
