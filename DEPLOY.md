# 🚀 Quick Deploy Guide

## Deploy Documentation to GitHub Pages

### One-Command Deploy

```bash
# Install docs dependencies
uv pip install -e ".[docs]"

# Deploy
mkdocs gh-deploy
```

**Done!** Your docs will be live at:
```
https://<your-username>.github.io/identifiability-guard/
```

### Before First Deploy

1. **Update `mkdocs.yml`**:
   ```yaml
   repo_url: https://github.com/YOUR_USERNAME/identifiability-guard
   ```

2. **Enable GitHub Pages** (if not auto-enabled):
   - Go to GitHub repo → Settings → Pages
   - Source: `gh-pages` branch, `/ (root)` folder
   - Save

### Auto-Deploy on Every Push

Create `.github/workflows/deploy-docs.yml`:

```yaml
name: Deploy Docs
on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - run: echo "$HOME/.cargo/bin" >> $GITHUB_PATH
      - run: uv pip install --system -e ".[docs]"
      - run: mkdocs gh-deploy --force
```

**Now docs deploy automatically!** ✨

See [GITHUB_PAGES_DEPLOYMENT.md](docs/GITHUB_PAGES_DEPLOYMENT.md) for full guide.
