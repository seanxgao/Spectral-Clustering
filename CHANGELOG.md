# Changelog

All notable changes to this project will be documented in this file.

## 2025-10-31

**API Changes:**
- Added type hints to function signatures (`eigen_decomposition`, `treebuilder`)

**Bug Fixes:**
- Fixed incorrect return value handling in `eigen_decomposition` call sites
- Corrected list length validation in `treebuilder` (changed from `.shape[0]` to `len()`)
- Improved error handling: `treebuilder` now returns `None` explicitly on failure

## 2025-01-25

- `eigen_decomposition` now returns both vector and eigenvalue
- Smart connectivity detection for large graphs (>5000 nodes)
- Added `find_connected_components` function
- Improved numerical precision for small matrices

