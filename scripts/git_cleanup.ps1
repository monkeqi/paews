# PAEWS Git Cleanup & Ship
# Run in PowerShell from C:\Users\josep\Documents\paews
# =====================================================

# 1. Remove old handoff versions (keep only the latest)
git rm PAEWS_HANDOFF.md
git rm PAEWS_HANDOFF_v2.md
git rm PAEWS_HANDOFF_v3.md
git rm PAEWS_HANDOFF_v4.md
git rm PAEWS_HANDOFF_v5.md
git rm PAEWS_SESSION_UPDATE.md

# 2. Remove PDFs that shouldn't be in the repo
git rm "1660-7804-1-PB.pdf"
git rm "Boletin 38-2 articulo5.pdf"
git rm "Boletin 38-2 articulo8.pdf"

# 3. Copy new files into repo root (do this BEFORE the commit)
# - Copy index.html from your Downloads into the repo root
# - Copy README.md from your Downloads into the repo root  
# - Rename PAEWS_HANDOFF_v9.md to PAEWS_HANDOFF.md (single current version)

# 4. Stage everything
git add -A

# 5. Commit and push
git commit -m "Ship v2: README, showcase page, repo cleanup. Sessions 16-19."
git push origin main

# 6. Enable GitHub Pages (do this in browser):
#    github.com/monkeqi/paews → Settings → Pages → Source: Deploy from branch → main → / (root) → Save
#    Your site will be live at: https://monkeqi.github.io/paews/
