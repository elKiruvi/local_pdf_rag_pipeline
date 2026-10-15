"""Create a newly authored, generic PDF fixture locally; never ship a corpus."""

from pathlib import Path

import pymupdf


def create_demo(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for heading, lines in [
        (
            "WORKSTATION STARTUP",
            [
                "Connect the display and power cable before starting the workstation.",
                "Press the power button once. Wait for the status light to turn green.",
                "Step | Action | Expected result",
                "1 | Connect power | Status light is off",
                "2 | Press button | Status light turns green",
            ],
        ),
        (
            "BACKUP SCHEDULE",
            [
                "Create a backup every Friday and store it on a removable drive.",
                "Check the backup by restoring one test file before disconnecting the drive.",
            ],
        ),
    ]:
        page = document.new_page()
        page.insert_text((72, 72), heading, fontsize=16)
        for number, line in enumerate(lines):
            page.insert_text((72, 110 + 25 * number), line, fontsize=10)
    document.set_metadata({"title": "Synthetic workstation handbook", "author": "", "subject": ""})
    document.save(path, garbage=4, deflate=True)
    document.close()
    return path
