# Documentation Setup Guide

This project uses **MkDocs Material** with **mkdocstrings** to automatically generate beautiful documentation from your code's docstrings.

## What We Have

### Auto-Generated API Docs
The API documentation is automatically extracted from your code's docstrings:
- `docs/api/dgp.md` - Auto-generates from `src/identifiability_guard/dgp/`
- `docs/api/encoders.md` - Auto-generates from `src/identifiability_guard/encoders/`
- `docs/api/metrics.md` - Auto-generates from `src/identifiability_guard/metrics/`
- `docs/api/evaluation.md` - Auto-generates from `src/identifiability_guard/evaluation/`

### Manual Documentation
Hand-written guides for users:
- `docs/index.md` - Homepage
- `docs/installation.md` - Installation instructions
- `docs/dgp.md` - DGP user guide
- `docs/encoders.md` - Encoder user guide
- `docs/metrics.md` - Metrics user guide
- `docs/examples.md` - Example usage patterns
- `docs/contributing.md` - Development guide

## Quick Start

```bash
# Install documentation tools
uv pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve

# Open browser to http://127.0.0.1:8000
```

## How It Works

### 1. MkDocs Configuration (`mkdocs.yml`)
Defines:
- Site metadata and theme
- Navigation structure
- Plugins (mkdocstrings for auto-docs)
- Markdown extensions (math, code highlighting)

### 2. Docstring Format (Google Style)
Your code uses Google-style docstrings which are automatically parsed:

```python
class MyClass:
    """
    Short one-line summary.

    Longer description with more details about what this class does
    and how to use it.

    Attributes:
        param1: Description of param1
        param2: Description of param2

    Example:
        >>> obj = MyClass(param1=5)
        >>> obj.method()
        42
    """

    def method(self, arg: int) -> float:
        """
        Short description of method.

        Args:
            arg: Description of argument

        Returns:
            Description of return value

        Raises:
            ValueError: When arg is negative
        """
        pass
```

### 3. Auto-Documentation Syntax
In markdown files, use `:::` to include auto-generated docs:

```markdown
## My Class

::: identifiability_guard.module.MyClass
    options:
      show_root_heading: true
      show_source: true
```

This pulls docstrings directly from your code!

## Building & Deployment

### Local Development
```bash
mkdocs serve  # Auto-reload on changes
```

### Build Static Site
```bash
mkdocs build  # Creates site/ directory
```

### Deploy to GitHub Pages
```bash
mkdocs gh-deploy  # Automatically deploys to gh-pages branch
```

### Deploy to Other Platforms
The `site/` directory contains static HTML that can be hosted anywhere:
- Netlify
- Vercel
- AWS S3
- Read the Docs

## Features

✅ **Automatic API extraction** from docstrings
✅ **Live reload** during development
✅ **Search functionality**
✅ **Math rendering** (LaTeX via MathJax)
✅ **Syntax highlighting** for code
✅ **Dark/light theme** toggle
✅ **Mobile responsive**
✅ **GitHub integration**

## Maintaining Documentation

### When Adding New Code
Just write good docstrings! The API docs will auto-update.

### When Adding New Features
1. Add docstrings to new classes/functions
2. Update relevant user guide (e.g., `docs/metrics.md`)
3. Add example to `docs/examples.md`
4. That's it!

### Testing Docs Locally
```bash
mkdocs serve
# Check http://127.0.0.1:8000
```

## Customization

### Theme Colors
Edit `mkdocs.yml`:
```yaml
theme:
  palette:
    primary: indigo  # Change color
```

### Navigation
Edit `nav:` section in `mkdocs.yml`

### Add Extensions
Add to `markdown_extensions:` in `mkdocs.yml`

## Troubleshooting

**Issue**: Import errors when building docs
```bash
# Solution: Install package in editable mode
uv pip install -e .
```

**Issue**: Docstrings not showing
- Check docstring format (must be Google-style)
- Verify import path in `:::` directive
- Check module is in `__all__` if using it

**Issue**: Math not rendering
- Verify `pymdownx.arithmatex` is enabled
- Check MathJax is loaded in `extra_javascript`
- Use `$$` for block math, `$` for inline

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material Theme](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Google Docstring Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
