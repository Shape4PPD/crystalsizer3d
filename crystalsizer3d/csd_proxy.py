import json
from typing import List, Optional

from crystalsizer3d import CSD_PROXY_PATH, logger
from crystalsizer3d.util.utils import is_main_thread


def _check_for_csd() -> bool:
    if not is_main_thread():
        return False
    try:
        import ccdc
        return True
    except Exception:
        return False


class CellStructure:
    def __init__(
            self,
            lattice_unit_cell: List[float],
            lattice_angles: List[float],
            point_group_symbol: str = '1',
    ):
        self.lattice_unit_cell = lattice_unit_cell
        self.lattice_angles = lattice_angles
        self.point_group_symbol = point_group_symbol


class CSDProxy:

    def __init__(self):
        self._init_reader()
        self._init_local_db()

    def load(self, crystal_id: str, use_cache: bool = True) -> CellStructure:
        """
        Load a crystal structure from the CSD database.
        """
        cs = None

        # Try to load it from the cache
        if use_cache:
            cs = self._load_from_local_db(crystal_id)
            if cs is not None:
                return cs

        # If not found, try to load it from the CSD database
        if cs is None and not self.csd_available:
            raise RuntimeError('CSD is not available.')
        cs = self._load_from_csd(crystal_id)

        # Save to the cache
        self.local_db[crystal_id] = {
            'lattice_unit_cell': cs.lattice_unit_cell,
            'lattice_angles': cs.lattice_angles,
            'point_group_symbol': cs.point_group_symbol
        }
        with open(CSD_PROXY_PATH, 'w') as f:
            json.dump(self.local_db, f, indent=4)

        return cs

    def _init_reader(self):
        """
        Try to initialize the CSD database reader.
        """
        self.csd_available = _check_for_csd()
        if not self.csd_available:
            return

        # Load the crystal template from the CSD database
        from ccdc.io import EntryReader
        self.reader = EntryReader()

    def _init_local_db(self):
        """
        Initialize the local database.
        """
        if not CSD_PROXY_PATH.exists():
            logger.info(f'Creating local CSD proxy database at {CSD_PROXY_PATH}')
            if not CSD_PROXY_PATH.parent.exists():
                CSD_PROXY_PATH.parent.mkdir(parents=True)
            with open(CSD_PROXY_PATH, 'w') as f:
                json.dump({}, f)
        with open(CSD_PROXY_PATH, 'r') as f:
            self.local_db = json.load(f)

    def _load_from_csd(self, crystal_id: str) -> Optional[CellStructure]:
        """
        Load a crystal from the CSD database.
        """
        assert self.csd_available, 'CSD is not available.'
        from ccdc.crystal import Crystal as CSD_Crystal
        csd_crystal: CSD_Crystal = self.reader.crystal(crystal_id)
        return CellStructure(
            lattice_unit_cell=[
                csd_crystal.cell_lengths[0],
                csd_crystal.cell_lengths[1],
                csd_crystal.cell_lengths[2]
            ],
            lattice_angles=[
                csd_crystal.cell_angles[0],
                csd_crystal.cell_angles[1],
                csd_crystal.cell_angles[2]
            ],
            point_group_symbol='222'  # crystal.spacegroup_symbol
        )

    def _load_from_local_db(self, crystal_id: str) -> Optional[CellStructure]:
        """
        Load a crystal from the local database.
        """
        if crystal_id in self.local_db:
            return CellStructure(**self.local_db[crystal_id])
        return None
