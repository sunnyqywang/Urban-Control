import glob
import os
os.chdir("/home/gridsan/qwang/urban-control/")
import numpy as np
import pandas as pd

files = glob.glob("data/landuse_overlay/*_[0-9].csv")
filtered_files = glob.glob("data/landuse_overlay/*_filtered.csv")
d = {}
for f in files:
    if f.replace(".csv", "_filtered.csv") not in filtered_files:
        city, offset_right, offset_down = f.split('/')[-1].split('.')[0].split('_')

        df = pd.read_csv(f)

        landuse_cols = [c for c in df.columns if 'area_m2' in c]

        landuse_cols.append('parking')
        
        # calculate landuse cover
        df['total_landuse_cover'] = 0
        for c in landuse_cols:
            df['total_landuse_cover'] += df[c].fillna(0)
        df['total_landuse_cover'] += df['water'].fillna(0)
        
        # normalize if >1
        df.loc[df['total_landuse_cover']>1, landuse_cols] /= df['total_landuse_cover']
        
        # recalculate
        df['total_landuse_cover'] = 0
        for c in landuse_cols:
            df['total_landuse_cover'] += df[c].fillna(0)
        df['total_landuse_cover'] += df['water'].fillna(0)
        
        # Filter
        df = df[df['total_landuse_cover']>0.7]
        df['total_landuse_cover'] -= df['water'].fillna(0)
        df = df[df['total_landuse_cover']>0.3]

        
        landuse_building_cols = ['area_m2_commercial',
        'area_m2_industrial',
        'area_m2_recreational',
        'area_m2_residential']

        df['building_landuse_denominator'] = 0
        for c in landuse_building_cols:
            df['building_landuse_denominator'] += df[c].fillna(0)
        df['building_landuse_denominator'] = np.where(df['building_landuse_denominator']==0, np.nan, df['building_landuse_denominator'])

        df['building_density'] = df['tot_building_footprint']/df['building_landuse_denominator']
        df['building_density'] = np.where(df['building_density'] <= 1, df['building_density'], np.nan)
        
        if 'city' not in d.keys():
            d[city] = df['building_density'].dropna().tolist()
        else:
            d[city].append(df['building_density'].dropna().tolist())
            
        df.to_csv(f"data/landuse_overlay/{city}_{offset_right}_{offset_down}_filtered.csv", index=False)

        print(city, offset_right, offset_down, len(df))
        