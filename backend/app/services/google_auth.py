"""OAuth2 credential management for Google APIs (multi-account)."""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from app.settings import get_settings

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

SCOPES_READONLY = [
    "https://www.googleapis.com/auth/gmail.readonly",
]

_TOKEN_PREFIX = ".google_token"


def _token_path(account: str = "primary") -> Path:
    root = Path(get_settings().project_root)
    if account == "primary":
        return root / f"{_TOKEN_PREFIX}.json"
    return root / f"{_TOKEN_PREFIX}_{account}.json"


def get_credentials(account: str = "primary") -> Credentials:
    """Load saved credentials for an account, auto-refresh if expired.

    Raises FileNotFoundError if no token file exists (run oauth setup first).
    """
    path = _token_path(account)
    if not path.exists():
        raise FileNotFoundError(
            f"No Google token found for account '{account}' at {path}. "
            f"Run: uv run scripts/google_oauth_setup.py --account {account}"
        )

    scopes = _scopes_for_account(account)
    creds = Credentials.from_authorized_user_file(str(path), scopes)

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        path.write_text(creds.to_json())

    return creds


def run_oauth_flow(account: str = "primary", readonly: bool = False) -> Credentials:
    """Interactive browser-based OAuth2 consent flow (one-time setup)."""
    settings = get_settings()

    scopes = SCOPES_READONLY if readonly else SCOPES

    client_config = {
        "installed": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uris": [settings.google_redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, scopes)
    creds = flow.run_local_server(port=8642, open_browser=True)

    # Save token for future use
    path = _token_path(account)
    path.write_text(creds.to_json())
    print(f"Token saved to {path}")

    return creds


def check_auth(account: str = "primary") -> dict:
    """Check if Google auth is configured and valid for an account."""
    try:
        creds = get_credentials(account)
        return {"authenticated": True, "valid": creds.valid, "expired": creds.expired, "account": account}
    except FileNotFoundError:
        return {
            "authenticated": False,
            "account": account,
            "error": f"No token file. Run: uv run scripts/google_oauth_setup.py --account {account}",
        }
    except Exception as e:
        return {"authenticated": False, "account": account, "error": str(e)}


def list_accounts() -> list[dict]:
    """Scan for configured Google accounts and their status."""
    root = Path(get_settings().project_root)
    settings = get_settings()
    readonly_names = {name.strip().lower() for name in settings.gmail_readonly_accounts.split(",") if name.strip()}

    accounts = []
    for token_file in sorted(root.glob(f"{_TOKEN_PREFIX}*.json")):
        name = token_file.stem
        if name == _TOKEN_PREFIX:
            account_name = "primary"
        else:
            # .google_token_work.json → "work"
            account_name = name[len(_TOKEN_PREFIX) + 1 :]

        is_readonly = account_name in readonly_names
        status = check_auth(account_name)
        email = ""
        if status.get("authenticated"):
            email = _get_email_address(account_name)
        accounts.append(
            {
                "account": account_name,
                "email": email,
                "readonly": is_readonly,
                "authenticated": status.get("authenticated", False),
                "token_file": token_file.name,
            }
        )

    return accounts


def _get_email_address(account: str) -> str:
    """Fetch the authenticated user's email address from Gmail API."""
    try:
        from googleapiclient.discovery import build

        creds = get_credentials(account)
        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        return profile.get("emailAddress", "")
    except Exception:
        return ""


def _scopes_for_account(account: str) -> list[str]:
    """Return the appropriate scopes for an account."""
    settings = get_settings()
    readonly_names = {name.strip().lower() for name in settings.gmail_readonly_accounts.split(",") if name.strip()}
    if account.lower() in readonly_names:
        return SCOPES_READONLY
    return SCOPES
