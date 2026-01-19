"""
Dashboard Application

Entry point: sets up sys.path, imports DashboardUI, and runs the Streamlit app.
"""

import os
import sys

# Add src and project root to path for imports
_current = os.path.dirname(os.path.abspath(__file__))
if _current not in sys.path:
    sys.path.insert(0, _current)
_parent = os.path.dirname(_current)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    from dashboard_ui import DashboardUI
except ImportError:
    from .dashboard_ui import DashboardUI  # type: ignore


if __name__ == "__main__":
    DashboardUI().run()
