# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Proxima Agent, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email security concerns to: security@proxima-project.io
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution Target**: Within 30 days for critical issues

## Security Best Practices

When using Proxima Agent:

### API Keys

- Never commit API keys to version control
- Use environment variables or the secure keyring
- Rotate keys periodically

### LLM Usage

- Always review consent prompts before confirming
- Prefer local LLM inference for sensitive data
- Be aware that remote LLMs may log prompts

### Docker

- Use specific version tags, not `latest` in production
- Run as non-root user (default in our images)
- Keep images updated for security patches

### Configuration

- Protect `~/.proxima/config.yaml` with appropriate permissions
- Review `force` operations carefully
- Enable consent requirements in production

## Dependencies

We monitor dependencies for known vulnerabilities using:

- GitHub Dependabot
- Safety (Python package auditing)

## Disclosure Policy

After a vulnerability is fixed:

1. We will publish a security advisory
2. Credit will be given to the reporter (if desired)
3. Users will be notified through release notes
