import os
import json
import sapien.core as sapien
import sapien.asset

SAPIEN_TOKEN = "YOUR_SAPIEN_TOKEN"
BASE_DIR = "/cluster/53/jankula/datasets/partnet_mobility"

# Initialize a headless SAPIEN scene to calculate dimensions
engine = sapien.Engine()
scene = engine.create_scene()
loader = scene.create_urdf_loader()

# Define the list of Object IDs you want to process
# (You can replace this array with the full range of PartNet IDs)
object_ids = [179, 180, 181] 

compiled_dataset = {}

for obj_id in object_ids:
    print(f"--- Processing Object {obj_id} ---")
    try:
        # 1. Download the minimal required URDF + structural files
        urdf_file = sapien.asset.download_partnet_mobility(
            model_id=obj_id, 
            token=SAPIEN_TOKEN, 
            directory=BASE_DIR
        )
        
        obj_dir = os.path.dirname(urdf_file)
        
        # 2. Extract Object Category (from the meta JSON file)
        meta_json_path = os.path.join(obj_dir, "meta.json")
        object_category = "Unknown"
        if os.path.exists(meta_json_path):
            with open(meta_json_path, 'r') as f:
                meta_data = json.load(f)
                object_category = meta_data.get("model_cat", "Unknown")

        # 3. Extract Part Labels (from semantics.txt)
        part_labels = {}
        semantics_path = os.path.join(obj_dir, "semantics.txt")
        if os.path.exists(semantics_path):
            with open(semantics_path, 'r') as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        part_labels[parts[0]] = parts[1] # Maps string ID to class label

        # 4. Extract 3D Bounding Boxes via the active URDF structure
        articulation = loader.load(urdf_file)
        bounding_boxes = {}
        
        for link in articulation.get_links():
            link_id = link.get_name().split("_")[-1] # Extracts numeric ID from 'link_0'
            collision_mesh = link.get_first_collision_mesh()
            
            if collision_mesh is not None:
                min_bbox, max_bbox = collision_mesh.bounds
                center = ((min_bbox + max_bbox) / 2).tolist()
                extents = (max_bbox - min_bbox).tolist() # [width, depth, height]
                
                bounding_boxes[link_id] = {
                    "label": part_labels.get(link_id, "unknown_part"),
                    "center_xyz": center,
                    "box_extents": extents
                }

        # 5. Save exactly what you need to your global dictionary object
        compiled_dataset[obj_id] = {
            "object_category": object_category,
            "parts_data": bounding_boxes
        }
        
        # [CLEANUP] Remove the heavy textured_objs subfolder to reclaim space instantly
        # This keeps your storage free of unneeded mesh geometries
        textured_dir = os.path.join(obj_dir, "textured_objs")
        if os.path.exists(textured_dir):
            os.system(f"rm -rf {textured_dir}")
            print(f"Cleaned up heavy asset meshes for object {obj_id}.")

    except Exception as e:
        print(f"Failed parsing object {obj_id}: {e}")

# Save the final structured output dataset
output_file = "/cluster/53/jankula/datasets/clean_bounding_boxes.json"
with open(output_file, 'w') as f:
    json.dump(compiled_dataset, f, indent=4)

print(f"\nAll done! Your custom lightweight dataset is saved at: {output_file}")