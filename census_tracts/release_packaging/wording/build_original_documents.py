"""Rebuild the original release's PDFs with the researcher definitions and pi_wording edits."""
import sys

import build_selected_documents as builder
import pi_wording

builder.researcher.configure(builder)
pi_wording.configure(builder)
pi_wording.configure_status(builder)
pi_wording.configure_tables(builder)
for path in (builder.build_codebook(), builder.build_guide()):
    print('Built', path)
