# TenderAI

TenderAI 是一个本地优先、开源的 AI 招投标助手。

当前项目处于早期开发阶段。

## Supported documents

- DOCX
- DOC (requires LibreOffice for conversion)
- PDF

DOC conversion requires LibreOffice. `LIBREOFFICE_PATH` can explicitly select its executable; on Windows, `soffice.com` is recommended for CLI use, while `soffice.exe` remains supported. Each conversion uses an isolated LibreOffice profile and has a default 180-second timeout.

On Windows, if LibreOffice is not on PATH, set `LIBREOFFICE_PATH` for the current PowerShell session:

```powershell
$env:LIBREOFFICE_PATH="<path-to-libreoffice>\program\soffice.com"
```

Maximum upload size: 500 MiB per file.

PDF technical drawings are preserved and detected, but OCR and vision understanding are not yet implemented.

## V0.2 development configuration

The current AI provider boundary uses an OpenAI-compatible API. Development environment variables are listed in `.env.example`; copy its values into a local `.env` only when needed. API keys must not be committed. Local compatible providers may use an empty API key.
