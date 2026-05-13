import pytest

def pytest_html_results_table_header(cells):
    cells.insert(2, "<th>KPI Value</th>")
    cells.insert(3, "<th>Threshold</th>")

def pytest_html_results_table_row(report, cells):
    cells.insert(2, f"<td>{getattr(report, 'kpi_value', '-')}</td>")
    cells.insert(3, f"<td>{getattr(report, 'threshold', '-')}</td>")

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == "call":
        props = dict(item.user_properties)
        report.kpi_value = props.get("Precision") or props.get("Recall") or \
                           props.get("F1 Score") or props.get("mAP") or \
                           props.get("MOTA") or props.get("MOTP") or \
                           props.get("ID_Switches") or "-"
        report.threshold = props.get("Threshold", "-")