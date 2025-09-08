import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from panopticapi.utils import rgb2id

class TripletDataset(Dataset):
    def __init__(self, json_file, coco_dir, split: str = None, support_rels: list = None, unsupport_rels: list = None):
        """
        Dataset ở mức triplet, mỗi sample = 1 (subject, predicate, object).
        - json_file: file psg.json
        - coco_dir: thư mục chứa images và panoptic maps
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        self.thing_cats = data["thing_classes"]
        self.stuff_cats = data["stuff_classes"]
        self.obj_cats = self.thing_cats + self.stuff_cats
        self.rel_cats = data["predicate_classes"]
        if support_rels is not None and unsupport_rels is not None:
            raise ValueError("support_rels and unsupport_rels cannot be both not None")
        if unsupport_rels is not None:
            self.support_rels = list(set(self.rel_cats) - set(unsupport_rels))
        else:
            self.support_rels = support_rels
        self.images = {d["image_id"]: d for d in data["data"]}
        self.coco_dir = coco_dir
        self.split = split
        self.found_rels = []
        # build danh sách triplets (image_id, s_idx, o_idx, rel_id)
        self.triplets = []
        max_trip_count = 0
        for img_id, d in self.images.items():
            if self.split is not None and self.split not in d["file_name"]:
                continue
            if len(d["relations"]) > max_trip_count:
                max_trip_count = len(d["relations"])
            for index, rel in enumerate(d["relations"]):
                s_idx, o_idx, rel_id = rel
                if self.support_rels is not None and self.rel_cats[rel_id] not in self.support_rels:
                    continue
                if self.rel_cats[rel_id] not in self.found_rels:
                    self.found_rels.append(self.rel_cats[rel_id])
                self.triplets.append((img_id, s_idx, o_idx, rel_id, index))
        print(f"max_trip_count: {max_trip_count}")
        print(f"found_rels: {len(self.found_rels)}")
        if self.support_rels is not None:
            print(f"support_rels: {len(self.support_rels)}")
        else:
            print(f"support_rels: {None}")
        print(f"rel_cats: {len(self.rel_cats)}")
        print(f"len(self.triplets): {len(self.triplets)}")
        # input()
    def overlay_masks_on_image(self, image, masks, labels, colors, alpha=0.5):
        """
        Overlay masks lên ảnh với colors và labels tương ứng
        
        Args:
            image: PIL Image gốc
            masks: list các mask numpy array (True/False)
            labels: list các label string
            colors: list các color tuple (R, G, B)
            alpha: độ trong suốt của mask (0-1)
        
        Returns:
            PIL Image đã overlay masks
        """
        # Convert PIL image to numpy array
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Tạo overlay image
        overlay = np.zeros_like(img_array)
        
        for mask, color in zip(masks, colors):
            # Resize mask nếu cần
            if mask.shape != (height, width):
                mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                mask = np.array(mask_pil) > 128
            
            # Apply color to mask areas
            overlay[mask] = color
        
        # Blend original image with overlay
        blended = img_array.copy()
        mask_areas = np.any(overlay > 0, axis=2)
        blended[mask_areas] = (
            (1 - alpha) * img_array[mask_areas] + 
            alpha * overlay[mask_areas]
        ).astype(np.uint8)
        
        # Convert back to PIL
        result_img = Image.fromarray(blended)
        
        # Add text labels if needed
        result_img = self.add_labels_to_image(result_img, masks, labels, colors)
        
        return result_img
    
    def add_labels_to_image(self, image, masks, labels, colors):
        """
        Thêm text labels vào ảnh tại vị trí của từng mask
        """
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for mask, label, color in zip(masks, labels, colors):
            # Find centroid of mask for label placement
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))
                
                # Draw text with background
                bbox = draw.textbbox((centroid_x, centroid_y), label, font=font)
                draw.rectangle(bbox, fill=(255, 255, 255, 128))
                draw.text((centroid_x, centroid_y), label, fill=color, font=font)
        
        return image
    
    def create_simple_overlay(self, image, masks, colors, alpha=0.3):
        """
        Tạo overlay đơn giản chỉ với màu sắc, không có labels
        """
        img_array = np.array(image)
        result = img_array.copy()
        
        for mask, color in zip(masks, colors):
            # Create colored overlay
            colored_mask = np.zeros_like(img_array)
            colored_mask[mask] = color
            
            # Blend with original image
            mask_3d = np.stack([mask] * 3, axis=-1)
            result = np.where(
                mask_3d,
                (1 - alpha) * result + alpha * colored_mask,
                result
            ).astype(np.uint8)
        
        return Image.fromarray(result)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        img_id, s_idx, o_idx, rel_id, index = self.triplets[idx]
        data = self.images[img_id]

        # load ảnh gốc
        img_path = os.path.join(self.coco_dir, data["file_name"])
        image = Image.open(img_path).convert("RGB")

        # load seg map
        seg_map_path = os.path.join(self.coco_dir, data["pan_seg_file_name"])
        seg_map = np.array(Image.open(seg_map_path).convert("RGB"))
        seg_map = rgb2id(seg_map)

        # tạo masks cho subject và object
        segments_info = data["segments_info"]
        s_mask = seg_map == segments_info[s_idx]["id"]
        o_mask = seg_map == segments_info[o_idx]["id"]

        # labels
        # s_label = self.obj_cats[segments_info[s_idx]["category_id"]]
        # o_label = self.obj_cats[segments_info[o_idx]["category_id"]]
        s_label = self.obj_cats[segments_info[s_idx]["category_id"]].split("-")[0]
        o_label = self.obj_cats[segments_info[o_idx]["category_id"]].split("-")[0]
        rel_label = self.rel_cats[rel_id]

        # overlay mask lên ảnh (thay thế Visualizer)
        masks = [s_mask, o_mask]
        labels = [s_label, o_label]
        colors = [(255, 0, 0), (0, 0, 255)]  # Red for subject, Blue for object
        
        # Option 1: Full overlay with labels
        viz_img = self.overlay_masks_on_image(image, masks, labels, colors, alpha=0.4)
        
        # Option 2: Simple overlay without labels (uncomment if preferred)
        # viz_img = self.create_simple_overlay(image, masks, colors, alpha=0.3)

        sample = {
            "triplet_id": f"{img_id}_{s_idx}_{o_idx}_{rel_id}_{index}",
            "image_id": img_id,
            "image": viz_img,
            "subject_class": s_label,
            "object_class": o_label,
            "gt_predicate": rel_label,
            # "subject_mask": s_mask,
            # "object_mask": o_mask,
            "subject_mask_color": "red",
            "object_mask_color": "blue",
            "gt_triplet": f"{s_label} {rel_label} {o_label}",
            "is_open_set": 'unset'
        }
        return sample


# Usage example
# if __name__ == "__main__":
#     json_file = 'data/psg/psg_val_test.json'
#     coco_dir = 'data/coco'
#     dataset = TripletDataset(json_file, coco_dir)

#     for sample in dataset:
#         print(f"Triplet ID: {sample['triplet_id']}")
#         print(f"Subject: {sample['subject_class']}")
#         print(f"Predicate: {sample['predicate']}")
#         print(f"Object: {sample['object_class']}")
#         #save image
#         sample['image'].save(f"visualize_example.png")
#         break