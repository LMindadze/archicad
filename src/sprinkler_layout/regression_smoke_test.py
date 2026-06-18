from __future__ import annotations

import sys
import unittest
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_SRC_ROOT = _THIS_FILE.parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sprinkler_layout.layout_engine import LayoutParameters, run_layout
from sprinkler_layout.schemas import DetectionDocument


def _fixture_detection_doc() -> DetectionDocument:
    return DetectionDocument.model_validate(
        {
            "schema_version": 2,
            "meta": None,
            "input_ifc": "fixture.ifc",
            "target_storey": "L1",
            "storeys_available": [],
            "detected_counts_on_target_storey": {},
            "slab_footprints": [],
            "unified_protected_floor_area": {
                "type": "Polygon",
                "exterior": [[0, 0], [24, 0], [24, 12], [0, 12], [0, 0]],
                "holes": [],
            },
            "columns": [],
            "columns_union": {
                "type": "Polygon",
                "exterior": [[11, 5], [13, 5], [13, 7], [11, 7], [11, 5]],
                "holes": [],
            },
            "stairs": [],
            "stairs_union": None,
            "walls_standard_case": [],
            "walls_standard_case_union": None,
            "walls_generic": [],
            "walls_generic_union": None,
            "walls_all_union": None,
            "generic_walls_failed_geometry": [],
            "spaces": [
                {
                    "ifc_id": 1,
                    "ifc_class": "IfcSpace",
                    "name": "A",
                    "storey": "L1",
                    "footprint": {
                        "type": "Polygon",
                        "exterior": [[0, 0], [12, 0], [12, 12], [0, 12], [0, 0]],
                        "holes": [],
                    },
                },
                {
                    "ifc_id": 2,
                    "ifc_class": "IfcSpace",
                    "name": "B",
                    "storey": "L1",
                    "footprint": {
                        "type": "Polygon",
                        "exterior": [[12, 0], [24, 0], [24, 12], [12, 12], [12, 0]],
                        "holes": [],
                    },
                },
            ],
            "spaces_union": None,
            "other_failures": {},
            "overall_floor_bounds": None,
            "candidate_axes": None,
            "suggested_trunk_line": None,
        }
    )


class LayoutRegressionSmokeTest(unittest.TestCase):
    def test_branch_grid_zone_outputs_present(self) -> None:
        doc = _fixture_detection_doc()
        params = LayoutParameters(layout_mode="branch_grid", candidate_layout_budget=10)
        result = run_layout(doc, params, show_progress=False)

        self.assertGreater(result.counts.zones, 0)
        self.assertGreater(result.counts.branch_lines, 0)
        self.assertGreater(result.counts.sprinkler_heads, 0)
        self.assertEqual(result.counts.jog_lines, len(result.geometries.jog_lines))
        self.assertTrue(result.geometries.zone_meta)
        self.assertIn("score_weights", result.quality_checks)


if __name__ == "__main__":
    unittest.main()
