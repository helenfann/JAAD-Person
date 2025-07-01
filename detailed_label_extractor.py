import os
import xml.etree.ElementTree as ET
import pandas as pd


#function to get the behaviours of a given pedestrian at a frame
def extract_frame_behaviors(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    frame_data = {}
    for box in root.findall('.//box'):
        # Get ID from the id tag
        id_attr = box.find('.//attribute[@name="id"]')


        # skip all un labelled IDs
        ped_id = id_attr.text.strip()  # Gets '0_11_56b' etc.
        if not ped_id.lower().endswith('b'):
            continue

        #extract behaviours
        frame_num = box.get('frame')
        behaviors = {}

        #customize this depedning on what attributes we want (maybe one hot encoding?)
        for attr in box.findall('.//attribute'):
            if attr.get('name') != 'old_id':
                behaviors[attr.get('name')] = attr.text


        # Aggregate behaviors for this frame
        if frame_num not in frame_data:
            frame_data[frame_num] = []
        frame_data[frame_num].append(behaviors)

    return frame_data

#Processes XML files and saves behaviors for 'b'-ending IDs.
def behaviors_to_aggregated_csv(xml_dir, output_csv):
    all_rows = []

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        frame_behaviors = extract_frame_behaviors(xml_path)

        for frame, behaviors in frame_behaviors.items():
            all_rows.append({
                'video': xml_file.replace('.xml', ''),
                'frame': frame,
                'behaviors': str(behaviors)  # Stringify list of behavior dicts
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} frames to {output_csv} (only IDs ending with 'b')")
    return df

# Example usage
xml_dir = "annotations"
output_csv = "frame_by_frame_behaviour_only_labelled.csv"
df = behaviors_to_aggregated_csv(xml_dir, output_csv)
