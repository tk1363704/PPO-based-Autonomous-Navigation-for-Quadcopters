from pathlib import Path
import airsim


def create_voxel_grid(output_path: Path):
    client = airsim.VehicleClient()
    center = airsim.Vector3r(0, 0, 0)
    res = 1
    client.simCreateVoxelGrid(center, 100, 100, 50, res, str(output_path.absolute()))
    print("voxel map generated!")
