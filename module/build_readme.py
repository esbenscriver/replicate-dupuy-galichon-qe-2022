"""
Build README.md from README.template.md by inserting content from output files.

This script reads README.template.md and replaces placeholders like {{output/file.md}}
with the actual content from those files.
"""

import re
from pathlib import Path


def load_file_content(file_path: str) -> str:
    """Load content from a file, return empty string if file doesn't exist."""
    path = Path(file_path)
    if path.exists():
        return path.read_text().strip()
    else:
        print(f"Warning: File {file_path} not found. Placeholder will be empty.")
        return ""


def build_readme(template_path: str = "output/README.template.md", output_path: str = "README.md"):
    """Build README.md from template by replacing placeholders with file content."""
    template = Path(template_path).read_text()

    # Find all placeholders in the format {{path/to/file.md}}
    pattern = r'\{\{([^}]+)\}\}'

    def replace_placeholder(match):
        file_path = match.group(1)
        content = load_file_content(file_path)
        return content

    # Replace all placeholders with file content
    readme_content = re.sub(pattern, replace_placeholder, template)

    # Write the final README.md
    Path(output_path).write_text(readme_content)
    print(f"✓ README.md built successfully from {template_path}")


if __name__ == "__main__":
    build_readme()
