import bpy, csv, math
from collections import defaultdict

R_E = 6371.0; H = 10.0
def wrap180(l): return ((l+180.0)%360.0)-180.0
def sph_to_xyz(lat_deg, lon_deg, h_km=H, R_E_km=R_E):
    lat = math.radians(lat_deg); lon = math.radians(wrap180(lon_deg)); r = R_E_km + h_km
    x = r*math.cos(lat)*math.cos(lon); y = r*math.cos(lat)*math.sin(lon); z = r*math.sin(lat)
    return (x, y, z)

csv_path = "/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf4/contours_10km.csv"
groups = defaultdict(list)
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        level = float(row["level"]); pid = int(row["path_id"]); idx = int(row["pt_idx"])
        lat = float(row["lat_deg"]); lon = float(row["lon_deg"])
        groups[(level, pid)].append((idx, lat, lon))

for (level, pid), pts in groups.items():
    pts.sort(key=lambda t: t[0])
    crv = bpy.data.curves.new(name=f"iso_{level:.2f}_{pid}", type='CURVE'); crv.dimensions='3D'
    poly = crv.splines.new('POLY'); poly.points.add(len(pts)-1)
    for i, (_, la, lo) in enumerate(pts):
        x,y,z = sph_to_xyz(la, lo)
        poly.points[i].co = (x, y, z, 1.0)  # w=1
    obj = bpy.data.objects.new(crv.name, crv)
    bpy.context.collection.objects.link(obj)
    mat = bpy.data.materials.new(name=f"mat_{level:.2f}")
    hue = (level % 5.0) / 5.0
    mat.diffuse_color = (1.0, 0.9, 0.2, 1.0) if abs(level-2.0)<1e-6 else (0.2, 0.7, 0.9, 1.0)
    obj.data.materials.append(mat)
