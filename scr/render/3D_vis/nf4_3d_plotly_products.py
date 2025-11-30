# -*- coding: utf-8 -*-
# nf4_3d_plotly_products.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def load_volume(npz_path: str):
    z = np.load(npz_path)
    dose = z["dose_muSvph"] 
    lats = z["lat_deg"]     
    lons = z["lon_deg"]      
    hs   = z["h_km"]         
    return dose, lats, lons, hs

def flatten_grid(dose, lats, lons, hs):
    K,L,M = dose.shape
    LAT, LON, H = np.meshgrid(lats, lons, hs, indexing="ij")  
    LAT = LAT.transpose(2,0,1)  
    LON = LON.transpose(2,0,1)
    H   = H.transpose(2,0,1)
    return LON.ravel(), LAT.ravel(), H.ravel(), dose.ravel()

def fig_isosurfaces(dose, lats, lons, hs, levels=(5,8,10), cmin=0, cmax=22):
    X, Y, Z, V = flatten_grid(dose, lats, lons, hs)
    fig = go.Figure()
    for i, lv in enumerate(levels):
        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z, value=V,
            isomin=lv, isomax=lv, surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            cmin=cmin, cmax=cmax, colorscale="Viridis",
            showscale=(i == 0),
            colorbar=(dict(title="μSv/h", len=0.75) if i == 0 else None),
            name=f"iso={lv} μSv/h",
        ))
    fig.update_layout(
        title="3D Radiation Isosurfaces (lon, lat, height[km])",
        scene=dict(xaxis_title="Longitude (°)",
                   yaxis_title="Latitude (°)",
                   zaxis_title="Height (km)"),
        legend=dict(orientation="h", y=1.02)
    )
    return fig


def fig_slice_slider(dose, lats, lons, hs, cmin=0, cmax=22):
    L, M = len(lats), len(lons)
    lon_grid, lat_grid = np.meshgrid(lons, lats)  
    frames = []
    k0 = 0
    surf0 = go.Surface(
        x=lon_grid, y=lat_grid, z=np.full_like(lon_grid, hs[k0]),
        surfacecolor=dose[k0], cmin=cmin, cmax=cmax, colorscale="Viridis",
        colorbar=dict(title="μSv/h")
    )
    fig = go.Figure(data=[surf0])
    for k,h in enumerate(hs):
        frames.append(go.Frame(
            name=f"{k}",
            data=[go.Surface(
                x=lon_grid, y=lat_grid, z=np.full_like(lon_grid, h),
                surfacecolor=dose[k], cmin=cmin, cmax=cmax, colorscale="Viridis",
                showscale=False
            )],
            traces=[0]
        ))
    fig.update(frames=frames)
    fig.update_layout(
        title="Height-Sliced Colored Surface (slider = km)",
        scene=dict(
            xaxis_title="Longitude (°)",
            yaxis_title="Latitude (°)",
            zaxis_title="Height (km)"
        ),
        sliders=[{
            "steps":[{"method":"animate","args":[[f"{i}"],{"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}],
                      "label":f"{hs[i]:.1f} km"} for i in range(len(hs))],
            "x":0.05,"y":-0.07,"len":0.9,"pad":{"t":30},
            "currentvalue":{"prefix":"Height: "}
        }],
        updatemenus=[{
            "type":"buttons","showactive":False,"x":1.05,"y":1.15,
            "buttons":[
                {"label":"Play","method":"animate","args":[None,{"frame":{"duration":300,"redraw":True},"fromcurrent":True}]},
                {"label":"Pause","method":"animate","args":[[None],{"frame":{"duration":0,"redraw":False},"mode":"immediate"}]}
            ]
        }]
    )
    return fig

def latlonh_to_xyz(lat, lon, h_km, R_E=6371.0):
    lat_r = np.deg2rad(lat); lon_r = np.deg2rad(lon)
    r = R_E + h_km
    x = r * np.cos(lat_r) * np.cos(lon_r)
    y = r * np.cos(lat_r) * np.sin(lon_r)
    z = r * np.sin(lat_r)
    return x, y, z

def fig_globe_points(dose, lats, lons, hs, sel_heights=(10,14,18,22), decim_step=2, cmin=0, cmax=22):
    fig = go.Figure()
    lon_grid, lat_grid = np.meshgrid(lons, lats) 
    for h in sel_heights:
        k = int(np.argmin(np.abs(hs - h)))
        Z = dose[k] 
        lat_s = lat_grid[::decim_step, ::decim_step].ravel()
        lon_s = lon_grid[::decim_step, ::decim_step].ravel()
        val_s = Z[::decim_step, ::decim_step].ravel()
        x, y, z = latlonh_to_xyz(lat_s, lon_s, np.full_like(lat_s, hs[k]))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=2, color=val_s, colorscale="Viridis", cmin=cmin, cmax=cmax, opacity=0.8),
            name=f"{hs[k]:.0f} km"
        ))
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(-np.pi/2, np.pi/2, 100)
    TH, PH = np.meshgrid(theta, phi)
    R = 6371.0
    xe = R * np.cos(PH) * np.cos(TH)
    ye = R * np.cos(PH) * np.sin(TH)
    ze = R * np.sin(PH)
    fig.add_trace(go.Surface(x=xe, y=ye, z=ze, showscale=False, opacity=0.2, colorscale=[[0,"#aaaaaa"],[1,"#cccccc"]], name="Earth"))
    fig.update_layout(
        title="Globe Point Cloud by Height (color = μSv/h)",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        legend=dict(orientation="h", y=1.05)
    )
    return fig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--isos", type=str, default="5,8,10")
    ap.add_argument("--globe-heights", type=str, default="10,14,18,22")
    ap.add_argument("--decim-step", type=int, default=2)
    ap.add_argument("--cmin", type=float, default=0.0)
    ap.add_argument("--cmax", type=float, default=22.0)
    args = ap.parse_args()

    dose, lats, lons, hs = load_volume(args.volume)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    isos = [float(x) for x in args.isos.split(",") if x]
    globe_heights = [float(x) for x in args.globe_heights.split(",") if x]

    fig1 = fig_isosurfaces(dose, lats, lons, hs, levels=isos, cmin=args.cmin, cmax=args.cmax)
    pio.write_html(fig1, file=str(outdir/"isosurfaces.html"), include_plotlyjs="cdn", auto_open=False)

    fig2 = fig_slice_slider(dose, lats, lons, hs, cmin=args.cmin, cmax=args.cmax)
    pio.write_html(fig2, file=str(outdir/"slice_slider.html"), include_plotlyjs="cdn", auto_open=False)


    fig3 = fig_globe_points(dose, lats, lons, hs, sel_heights=globe_heights,
                            decim_step=args.decim_step, cmin=args.cmin, cmax=args.cmax)
    pio.write_html(fig3, file=str(outdir/"globe_points.html"), include_plotlyjs="cdn", auto_open=False)

    meta = dict(
        isosurfaces=[float(x) for x in isos],
        globe_heights=globe_heights,
        cmin=args.cmin, cmax=args.cmax,
        K=int(dose.shape[0]), L=int(dose.shape[1]), M=int(dose.shape[2])
    )

    json.dump(meta, open(outdir/"3d_meta.json","w"), indent=2)
    print("[OK] Exported:", outdir)

if __name__ == "__main__":
    import json
    main()
