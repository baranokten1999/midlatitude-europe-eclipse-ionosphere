# Release checklist (GitHub → Zenodo)

1. Push this repository to GitHub (public).
2. On Zenodo: Profile → GitHub → Sync now → toggle this repo ON.
3. Create a GitHub Release (e.g., v1.0.0). Zenodo will mint a DOI for that release.
4. Paste the Zenodo *version DOI* into:
   - README.md (Citation section)
   - CITATION.cff (add `doi:` field)
   - your manuscript Data/Code availability section and references

Notes:
- Cite the *version DOI* (v1.0.0) for reproducibility.
- Use the concept DOI only as a secondary “all versions” pointer.
