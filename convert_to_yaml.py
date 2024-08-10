import pandas as pd
import numpy as np
df_rh = pd.read_csv('./test_dataset/using_phone/data/grt.csv')

import yaml

sorted_GT_tr=df_rh.sort_values(by=['personID', 'frame'])
num_people = sorted_GT_tr['personID'].unique()


tracklets = {}
track_id = 0
for p_ID in num_people:
    P_track = sorted_GT_tr.loc[sorted_GT_tr['personID']==p_ID]
    track_id = track_id + 1
    subject_id = p_ID
    start = P_track['frame'].iloc[0]
    end = P_track['frame'].iloc[-1]
    frames = {}

    for _id in P_track['frame']:   
        for fr_id in range(start, end+1): 
            if fr_id == _id:
                P_track_fr = P_track.loc[P_track['frame']==_id] 
                frames[_id] = [P_track_fr['xmin'].iloc[0], P_track_fr['ymin'].iloc[0], P_track_fr['xmax'].iloc[0], P_track_fr['ymax'].iloc[0]]

            elif fr_id >= _id + 1:            
                frames[fr_id] = []  
            else:
                continue
                
    tracklets[track_id] = {
        'subject_id': subject_id,
        'start': start,
        'end': end,
        'bbox': None,
        'frames': frames,
    } 
    
tracklets2 = {}
for i in tracklets:
    tracklets2[i] = {}
    tracklets2[i]['subject_id'] = int(tracklets[i]['subject_id'])
    tracklets2[i]['start'] = int(tracklets[i]['start'])
    tracklets2[i]['end'] = int(tracklets[i]['end'])
    all_bboxes = np.stack([bbox for bbox in tracklets[i]['frames'].values()
                               if len(bbox) != 0])
    tracklets2[i]['bbox'] = [all_bboxes[:, 0].min().item(),
                                       all_bboxes[:, 1].min().item(),
                                       all_bboxes[:, 2].max().item(),
                                       all_bboxes[:, 3].max().item()]
   
    tracklets2[i]['frames'] = {}
    for fr, coords in tracklets[i]['frames'].items():
        coords_ = [int(x) for x in coords]
        tracklets2[i]['frames'][fr] = coords_


tracking_output_file = './test_dataset/using_phone/data/grt.yaml'
with open(tracking_output_file, 'w') as of:
  yaml.safe_dump(tracklets2, of, indent=4, default_flow_style=None, sort_keys=False)