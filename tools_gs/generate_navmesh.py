#!/usr/bin/env python3

"""
Script to generate navigation mesh (NavMesh) from a collision mesh using Habitat-sim.
The NavMesh is used for pathfinding and navigation constraints in simulation environments.

Usage:
    python generate_navmesh.py --input path/to/mesh.glb --output path/to/output.navmesh
    
    # With custom parameters:
    python generate_navmesh.py --input mesh.glb --output mesh.navmesh \\
        --agent-height 1.5 --agent-radius 0.2 --cell-size 0.05
"""

import argparse
import os
import sys

import habitat_sim


def generate_navmesh(
    input_mesh_path,
    output_navmesh_path,
    navmesh_settings=None,
):
    """
    Generate a NavMesh from an input collision mesh and save it to disk.
    
    Args:
        input_mesh_path: Path to the input collision mesh file (e.g., .glb, .gltf, .obj, .ply)
        output_navmesh_path: Path where the generated .navmesh file will be saved
        navmesh_settings: NavMeshSettings object with custom parameters. If None, uses defaults.
    
    Returns:
        bool: True if NavMesh generation and saving succeeded, False otherwise
    """
    # Validate input file exists
    if not os.path.exists(input_mesh_path):
        print(f"Error: Input mesh file not found: {input_mesh_path}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_navmesh_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error: Failed to create output directory: {e}")
            return False
    
    # Configure simulator
    print(f"Loading mesh: {input_mesh_path}")
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = input_mesh_path
    sim_cfg.create_renderer = False  # No rendering needed for NavMesh generation
    sim_cfg.load_semantic_mesh = False
    
    # Create a minimal agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    # Create simulator configuration
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    # Initialize simulator
    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"Error: Failed to initialize simulator: {e}")
        return False
    
    # Use provided settings or create default settings
    if navmesh_settings is None:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
    
    # Print NavMesh settings
    print("\nNavMesh Settings:")
    print(f"  Cell size: {navmesh_settings.cell_size}")
    print(f"  Cell height: {navmesh_settings.cell_height}")
    print(f"  Agent height: {navmesh_settings.agent_height}")
    print(f"  Agent radius: {navmesh_settings.agent_radius}")
    print(f"  Agent max climb: {navmesh_settings.agent_max_climb}")
    print(f"  Agent max slope: {navmesh_settings.agent_max_slope}")
    print(f"  Region min size: {navmesh_settings.region_min_size}")
    print(f"  Region merge size: {navmesh_settings.region_merge_size}")
    print(f"  Edge max length: {navmesh_settings.edge_max_len}")
    print(f"  Edge max error: {navmesh_settings.edge_max_error}")
    print(f"  Verts per poly: {navmesh_settings.verts_per_poly}")
    print(f"  Detail sample dist: {navmesh_settings.detail_sample_dist}")
    print(f"  Detail sample max error: {navmesh_settings.detail_sample_max_error}")
    print(f"  Filter low hanging obstacles: {navmesh_settings.filter_low_hanging_obstacles}")
    print(f"  Filter ledge spans: {navmesh_settings.filter_ledge_spans}")
    print(f"  Filter walkable low height spans: {navmesh_settings.filter_walkable_low_height_spans}")
    
    # Compute NavMesh
    print("\nComputing NavMesh...")
    try:
        navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    except Exception as e:
        print(f"Error: NavMesh computation failed: {e}")
        sim.close()
        return False
    
    if not navmesh_success:
        print("Error: Failed to compute NavMesh. Try different parameters?")
        sim.close()
        return False
    
    print("NavMesh computed successfully!")
    
    # Check if PathFinder is loaded
    if not sim.pathfinder.is_loaded:
        print("Error: PathFinder not loaded after NavMesh computation")
        sim.close()
        return False
    
    # Print NavMesh statistics
    print(f"\nNavMesh Statistics:")
    print(f"  Navigable area: {sim.pathfinder.navigable_area:.2f} m²")
    bounds = sim.pathfinder.get_bounds()
    print(f"  Bounds: min={bounds[0]}, max={bounds[1]}")
    
    # Save NavMesh
    print(f"\nSaving NavMesh to: {output_navmesh_path}")
    try:
        sim.pathfinder.save_nav_mesh(output_navmesh_path)
        print("NavMesh saved successfully!")
    except Exception as e:
        print(f"Error: Failed to save NavMesh: {e}")
        sim.close()
        return False
    
    # Clean up
    sim.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate NavMesh from collision mesh using Habitat-sim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Generate NavMesh with default parameters:
  python generate_navmesh.py --input mesh.glb --output mesh.navmesh
  
  # Generate NavMesh with custom agent parameters:
  python generate_navmesh.py --input mesh.glb --output mesh.navmesh \\
      --agent-height 1.8 --agent-radius 0.25
  
  # Generate NavMesh with custom voxelization parameters:
  python generate_navmesh.py --input mesh.glb --output mesh.navmesh \\
      --cell-size 0.1 --cell-height 0.3
  
  # Generate NavMesh with all custom parameters:
  python generate_navmesh.py --input mesh.glb --output mesh.navmesh \\
      --cell-size 0.05 --cell-height 0.2 \\
      --agent-height 1.5 --agent-radius 0.2 \\
      --agent-max-climb 0.3 --agent-max-slope 50.0
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', 
        required=True, 
        help='Input collision mesh file path (e.g., .glb, .gltf, .obj, .ply)'
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help='Output NavMesh file path (e.g., .navmesh)'
    )
    
    # Voxelization parameters
    parser.add_argument(
        '--cell-size', 
        type=float, 
        default=None,
        help='XZ-plane voxel dimensions (default: 0.05). Smaller values = better accuracy but slower computation'
    )
    parser.add_argument(
        '--cell-height', 
        type=float, 
        default=None,
        help='Y-axis voxel dimension (default: 0.2). Smaller values = better accuracy but slower computation'
    )
    
    # Agent parameters
    parser.add_argument(
        '--agent-height', 
        type=float, 
        default=None,
        help='Height of the agent in meters (default: 1.5). Used to cull navigable cells with obstructions'
    )
    parser.add_argument(
        '--agent-radius', 
        type=float, 
        default=None,
        help='Radius of the agent in meters (default: 0.1). Used to erode/shrink the computed heightfield'
    )
    parser.add_argument(
        '--agent-max-climb', 
        type=float, 
        default=None,
        help='Maximum ledge height that is considered traversable in meters (default: 0.2)'
    )
    parser.add_argument(
        '--agent-max-slope', 
        type=float, 
        default=None,
        help='Maximum slope that is considered navigable in degrees (default: 45.0, range: 0-85)'
    )
    
    # Navigable area filtering options
    parser.add_argument(
        '--no-filter-low-hanging-obstacles',
        action='store_true',
        help='Disable filtering of low hanging obstacles (default: enabled)'
    )
    parser.add_argument(
        '--no-filter-ledge-spans',
        action='store_true',
        help='Disable filtering of ledge spans (default: enabled)'
    )
    parser.add_argument(
        '--no-filter-walkable-low-height-spans',
        action='store_true',
        help='Disable filtering of walkable low height spans (default: enabled)'
    )
    
    # Detail mesh generation parameters
    parser.add_argument(
        '--region-min-size', 
        type=int, 
        default=None,
        help='Minimum number of cells allowed to form isolated island areas (default: 20)'
    )
    parser.add_argument(
        '--region-merge-size', 
        type=int, 
        default=None,
        help='Any 2-D regions with smaller span will be merged with larger regions if possible (default: 20)'
    )
    parser.add_argument(
        '--edge-max-len', 
        type=float, 
        default=None,
        help='Maximum allowed length for contour edges along mesh border (default: 12.0)'
    )
    parser.add_argument(
        '--edge-max-error', 
        type=float, 
        default=None,
        help='Maximum distance a contour edge can deviate from raw contour (default: 1.3)'
    )
    parser.add_argument(
        '--verts-per-poly', 
        type=float, 
        default=None,
        help='Maximum vertices allowed for polygons (default: 6.0, range: 3-6)'
    )
    parser.add_argument(
        '--detail-sample-dist', 
        type=float, 
        default=None,
        help='Sampling distance for generating detail mesh (default: 6.0)'
    )
    parser.add_argument(
        '--detail-sample-max-error', 
        type=float, 
        default=None,
        help='Maximum distance detail mesh surface can deviate from heightfield (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Create NavMeshSettings
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    
    # Apply custom parameters if specified
    if args.cell_size is not None:
        navmesh_settings.cell_size = args.cell_size
    if args.cell_height is not None:
        navmesh_settings.cell_height = args.cell_height
    if args.agent_height is not None:
        navmesh_settings.agent_height = args.agent_height
    if args.agent_radius is not None:
        navmesh_settings.agent_radius = args.agent_radius
    if args.agent_max_climb is not None:
        navmesh_settings.agent_max_climb = args.agent_max_climb
    if args.agent_max_slope is not None:
        if args.agent_max_slope < 0 or args.agent_max_slope >= 85:
            print("Error: agent-max-slope must be in range [0, 85)")
            sys.exit(1)
        navmesh_settings.agent_max_slope = args.agent_max_slope
    
    # Apply filtering options
    if args.no_filter_low_hanging_obstacles:
        navmesh_settings.filter_low_hanging_obstacles = False
    if args.no_filter_ledge_spans:
        navmesh_settings.filter_ledge_spans = False
    if args.no_filter_walkable_low_height_spans:
        navmesh_settings.filter_walkable_low_height_spans = False
    
    # Apply detail mesh parameters
    if args.region_min_size is not None:
        navmesh_settings.region_min_size = args.region_min_size
    if args.region_merge_size is not None:
        navmesh_settings.region_merge_size = args.region_merge_size
    if args.edge_max_len is not None:
        navmesh_settings.edge_max_len = args.edge_max_len
    if args.edge_max_error is not None:
        navmesh_settings.edge_max_error = args.edge_max_error
    if args.verts_per_poly is not None:
        if args.verts_per_poly < 3 or args.verts_per_poly > 6:
            print("Error: verts-per-poly must be in range [3, 6]")
            sys.exit(1)
        navmesh_settings.verts_per_poly = args.verts_per_poly
    if args.detail_sample_dist is not None:
        navmesh_settings.detail_sample_dist = args.detail_sample_dist
    if args.detail_sample_max_error is not None:
        navmesh_settings.detail_sample_max_error = args.detail_sample_max_error
    
    # Generate NavMesh
    success = generate_navmesh(args.input, args.output, navmesh_settings)
    if not success:
        print("Error: Failed to generate NavMesh")
        sys.exit(1)

if __name__ == '__main__':
    main()

