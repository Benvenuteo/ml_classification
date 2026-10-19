"""Check that delivered notebook cells contain real executed rich output."""

import nbformat
import pytest

from air_quality.reporting import execute_notebook_cells


def test_notebook_execution_preserves_state_and_captures_rich_output():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("value = 41\nprint(value)"),
            nbformat.v4.new_code_cell(
                "from IPython.display import display, HTML\ndisplay(HTML(f'<b>{value + 1}</b>'))"
            ),
        ]
    )
    execute_notebook_cells(notebook)
    nbformat.validate(notebook)
    assert "41" in notebook.cells[0].outputs[0].text
    assert any(
        output.get("data", {}).get("text/html") == "<b>42</b>"
        for output in notebook.cells[1].outputs
    )
    assert notebook.cells[1].execution_count > notebook.cells[0].execution_count


def test_notebook_execution_errors_abort_generation():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("raise ValueError('invalid analysis')"),
        ]
    )
    with pytest.raises(ValueError, match="invalid analysis"):
        execute_notebook_cells(notebook)
