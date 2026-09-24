# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | Yes |
| Older releases | No |

## Reporting a Vulnerability

If you discover a security vulnerability in sushi, please report it privately:

1. **Do not** open a public issue
2. Open a private security advisory at
   [github.com/beamivalice/sushi/security/advisories/new](https://github.com/beamivalice/sushi/security/advisories/new) with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

For anything that is not a vulnerability, use [GitHub issues](https://github.com/beamivalice/sushi/issues).

## Security Model

sushi is designed as a **local inference server** running on a single machine. It is not designed for production
deployment or untrusted network exposure.

### Authentication
- **Off by default.** With no key set, every request is served.
- **`--api-key <token>`** (or `--api-key-env <VAR>`, which keeps the key out of the process table) requires the key on
  every request from another machine: `Authorization: Bearer`, `x-api-key`, HTTP Basic (key = password) or
  `?api_key=`. `GET /health` and CORS preflight stay open.
- Loopback requests are trusted unless **`--api-key-strict`** is also given.

### Recommendations
- The default bind is `0.0.0.0` (every interface, with a warning at startup). Pass `--host 127.0.0.1` unless you mean to
  serve the network, and set `--api-key` if you do.
- Only load models from trusted sources
- Do not run the server as root
