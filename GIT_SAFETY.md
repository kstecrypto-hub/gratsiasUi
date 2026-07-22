# Git safety notes

This repository copy intentionally excludes local credentials, recordings,
transcripts, databases, caches, dependencies, and build output.

## Local setup

1. Copy `.env.example` to `.env`.
2. Fill in secrets only in the local `.env`, or enter supported credentials in
   the application's Settings page.
3. Never commit `.env`, Docker volume exports, `storage/`, or `work/` contents.

The `.env.example` file is safe to commit: credential fields are blank and it
contains only documented defaults and placeholders.

Credentials entered through the UI are runtime data stored in the configured
database/Redis services. Do not add database dumps or Docker volume exports to
the repository.

Before publishing, check staged files with:

```powershell
git status --short
git diff --cached --name-only
```

If a real credential is ever committed, remove it from the repository history
and rotate it immediately.
