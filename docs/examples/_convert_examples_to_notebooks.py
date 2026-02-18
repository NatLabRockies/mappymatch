from pathlib import Path

import nbformat


def script_to_notebook(script_path: Path, notebook_path: Path) -> None:
    # Read the script
    with open(script_path, "r") as script_file:
        lines = script_file.readlines()

    notebook = nbformat.v4.new_notebook()
    current_code_block: list[str] = []
    current_markdown_block: list[str] = []

    def add_code_cell(block: list[str]) -> None:
        if block:
            notebook.cells.append(nbformat.v4.new_code_cell("".join(block)))

    def add_markdown_cell(block: list[str]) -> None:
        if block:
            notebook.cells.append(nbformat.v4.new_markdown_cell("".join(block).strip()))

    # markdown cells will be enclosed in triple quotes
    # the remaining cells will be code cells
    in_markdown = False
    in_main = False
    for line in lines:
        # strip any ipython code blocks
        if line.strip().startswith("# %%"):
            continue
        # strip def main(): line and track indentation state
        if line.strip() == "def main():":
            in_main = True
            continue
        # strip if __name__ == "__main__": and main() call
        if line.strip() in (
            'if __name__ == "__main__":',
            "main()",
        ):
            continue
        # dedent lines inside def main()
        if in_main and not line.strip().startswith('"""'):
            line = line[4:] if line.startswith("    ") else line
        if line.strip().startswith('"""'):
            # handle triple-quote toggle; dedent the """ line itself if inside main
            if in_main and line.startswith("    "):
                line = line[4:]
            in_markdown = not in_markdown
            if in_markdown:
                add_code_cell(current_code_block)
                current_code_block = []
            else:
                add_markdown_cell(current_markdown_block)
                current_markdown_block = []
        elif in_markdown:
            # dedent markdown content inside def main()
            if in_main and line.startswith("    "):
                line = line[4:]
            current_markdown_block.append(line)
        else:
            current_code_block.append(line)

    add_code_cell(current_code_block)
    add_markdown_cell(current_markdown_block)

    with open(notebook_path, "w") as notebook_file:
        nbformat.write(notebook, notebook_file)


if __name__ == "__main__":
    here = Path(__file__).parent
    examples = here.glob("*example.py")
    for example in examples:
        notebook = example.with_suffix(".ipynb")
        script_to_notebook(example, notebook)
        print(f"Converted {example} to {notebook}")
