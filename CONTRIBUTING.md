# Contributing to Zotero PDF Chat

Thank you for your interest in contributing to Zotero PDF Chat! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to create a welcoming and inclusive environment for everyone.

## Getting Started

1. Fork the repository
2. Clone your fork to your local machine
3. Set up your development environment following the instructions in `ENV_SETUP.md`
4. Create a new branch for your contribution

## Development Environment

Set up your development environment by:

1. Installing the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Setting up your environment variables as described in `ENV_SETUP.md`

## Making Changes

When making changes to the codebase:

1. Follow the existing code style and conventions
2. Write clear, descriptive commit messages
3. Keep your changes focused on a single issue or feature
4. Add tests for your changes when applicable
5. Update documentation as necessary

## Pull Request Process

1. Update your fork to include the latest changes from the main repository
2. Push your changes to your fork
3. Submit a pull request from your branch to the main repository
4. Clearly describe the changes and their purpose in your pull request description
5. Be responsive to feedback and be prepared to make additional changes if requested

## Important Security Notes

1. **Never commit sensitive information** such as API keys, personal data, or credentials
2. **Do not commit the contents of the `vector_db` directory**, which contains user-specific data
3. **Always ensure your `.env` file is excluded from git** (it should be listed in `.gitignore`)

## Testing

Before submitting your changes, make sure to:

1. Test your changes thoroughly
2. Ensure that all existing tests pass
3. Add new tests as appropriate for new features or bug fixes

## Documentation

When adding new features or making significant changes:

1. Update the README.md file as needed
2. Add or update docstrings for any new or modified functions, classes, or methods
3. Consider adding examples or usage instructions for complex features

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (see `LICENSE` file).

Thank you for your contributions! 