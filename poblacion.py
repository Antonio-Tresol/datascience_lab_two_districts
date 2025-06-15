import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""#<center>Construcción del Grafo Poblacional de Costa Rica a nivel Distrital</center>""")
    return


@app.cell
def _():
    import pandas as pd
    import geopandas as gpd
    import marimo as mo
    import networkx as nx
    import matplotlib.pyplot as plt
    import plotly.express as px
    import altair as alt
    return gpd, mo, nx, pd, plt, px


@app.cell
def open_data(gpd, mo, pd):
    district_population_data = pd.read_excel(
        mo.notebook_dir() / "legacy_notebooks/data/df_distritos.xlsx"
    )
    district_population_data["Codigo"] = district_population_data["Codigo"].astype("str")

    shapefile_path = mo.notebook_dir() / "legacy_notebooks/data/UGED_MGN_2022/UGED_MGN_2022.shp"

    district_geospatial_data = gpd.read_file(shapefile_path)

    district_geospatial_data = district_geospatial_data[
        district_geospatial_data.geometry.notnull()
    ].copy()

    # These coordinates are measured in meters on a flat plane, projected from the WGS 84 ellipsoid using
    # the Transverse Mercator method, with parameters optimized specifically for Costa Rica as defined by the CR05 standard.
    district_geospatial_data = district_geospatial_data.set_crs(epsg=5367)
    return district_geospatial_data, district_population_data


@app.cell
def show_population_data(district_population_data):
    district_population_data
    return


@app.cell
def show_geospatial_data(district_geospatial_data):
    district_geospatial_data
    return


@app.cell
def join_data(district_geospatial_data, district_population_data):
    district_data = district_population_data.merge(
        district_geospatial_data, left_on="Codigo", right_on="COD_UGED", how="inner"
    )

    district_data = district_data.drop(
        columns=[
            "COD_UGED",
            "NOMB_UGEC",
            "NOMB_UGED",
            "ID",
        ]
    )
    district_data = district_data.rename(
        columns={
            "NOMB_UGEP": "provincia",
            "Total": "poblacion_total",
        }
    )
    district_data = district_data.rename(
        columns={col: str(col).lower() for col in district_data.columns.to_list()}
    )
    district_data
    return (district_data,)


@app.cell
def _(district_data, gpd):
    geo_district_data = gpd.GeoDataFrame(district_data)
    return (geo_district_data,)


@app.cell
def _(mo):
    mo.md(r"""## Grafo de contiguidad""")
    return


@app.cell
def _(district_data, geo_district_data, nx):
    # Crear grafo de contigüidad
    G = nx.Graph()

    # Agregar nodos con población
    for idx, row in district_data.iterrows():
        G.add_node(idx, nombre=row['distrito'], poblacion=row['poblacion_total'])

    # Agregar aristas entre distritos contiguos
    for i, geom in district_data.geometry.items():
        vecinos = district_data[geo_district_data.geometry.touches(geom)].index
        for v in vecinos:
            if not G.has_edge(i, v):
                G.add_edge(i, v)
    return (G,)


@app.cell
def _(G, nx, plt):
    # Visualizar el grafo de contigüidad (sin base geográfica)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=10, with_labels=False)
    plt.title('Grafo de contigüidad entre distritos')
    plt.show()
    return


@app.cell
def _():
    ## TODO: VER ESTE GRAFO ENCIMA DEL MAPA DE CR
    return


@app.cell
def _(geo_district_data, px):
    def _():
        geo_district_data_viz = geo_district_data.to_crs(epsg=4326)

        # --- Interactive Visualization with Plotly Express ---

        # The choropleth_mapbox function does all the heavy lifting.
        fig = px.choropleth_mapbox(
            geo_district_data_viz,
            geojson=geo_district_data_viz.geometry,
            locations=geo_district_data_viz.index,
            color="poblacion_total",
            hover_name="distrito",
            hover_data={
                "provincia": True,
                "poblacion_total": ":,",
                "area_km2": ":.2f"
            },
            mapbox_style="carto-positron",
            center={"lat": 9.93, "lon": -84.08},
            zoom=7,
            opacity=0.7,
            title="Población de Distritos de Costa Rica"
        )

        # Customize the layout for a cleaner look
        fig.update_layout(
            margin={"r":0, "t":40, "l":0, "b":0},
            legend_title_text='Población'
        )

        # To show the figure in a notebook or script
        return fig.show()


    _()
    return


if __name__ == "__main__":
    app.run()
