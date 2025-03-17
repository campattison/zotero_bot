# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please follow these steps:

1. **Do not disclose the vulnerability publicly** via GitHub Issues, discussions, or pull requests
2. Email the project maintainer directly at [cameron.pattison@icloud.com]
3. Include as much information as possible about the vulnerability:
   - The steps to reproduce the issue
   - The potential impact of the vulnerability
   - Suggested fixes or mitigations if you have them

## Security Best Practices for Users

When using this application:

1. **Never share your .env file** or commit it to public repositories
2. **Keep your OpenAI API key secure** and don't share it with others
3. **Regularly rotate your API keys** if you suspect they may have been compromised
4. **Be aware of data privacy implications** when processing sensitive documents
5. **Use the included pre-commit hooks** to prevent accidentally committing sensitive information

## Security Implementation

This project implements several security measures:

1. **.gitignore configuration** to prevent committing sensitive files
2. **Environment variables** for storing sensitive configuration
3. **Pre-commit hooks** to detect potential security issues
4. **Local data processing** to avoid sending full documents to external services

## Dependency Security

We recommend:

1. **Regularly updating dependencies** to patch security vulnerabilities
2. **Using a dependency scanning tool** in your workflow
3. **Being cautious when adding new dependencies** to the project

## Third-Party Service Security

This application uses OpenAI's API. Be aware that:

1. Data sent to OpenAI is subject to their [privacy policy](https://openai.com/policies/privacy-policy)
2. Only the necessary text is sent to OpenAI, not your full PDF files
3. Consider the sensitivity of the documents you process with this tool 