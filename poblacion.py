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
    import openpyxl
    import pyarrow
    import plotly.graph_objects as go
    return go, gpd, mo, nx, pd, px


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
def make_geodataframe(district_data, gpd):
    geo_district_data = gpd.GeoDataFrame(district_data)
    return (geo_district_data,)


@app.cell
def _(mo):
    mo.md(r"""## Grafo de contiguidad""")
    return


@app.cell
def make_neighbor_graph(district_data, geo_district_data, nx):
    # Crear grafo de contigüidad
    G = nx.Graph()

    # Agregar nodos con población
    for idx, row in district_data.iterrows():
        G.add_node(idx, nombre=row["distrito"], poblacion=row["poblacion_total"])

    # Agregar aristas entre distritos contiguos
    for i, geom in district_data.geometry.items():
        vecinos = district_data[geo_district_data.geometry.touches(geom)].index
        for v in vecinos:
            if not G.has_edge(i, v):
                G.add_edge(i, v)
    return (G,)


@app.cell
def show_neighbor_graph(G, geo_district_data, go, px):
    def make_graph_map():
        # --- 1. Create the Base Map (with hover disabled on polygons) ---
        def create_base_map():
            geo_district_data_viz = geo_district_data.to_crs(epsg=4326)

            fig = px.choropleth_map(
                geo_district_data_viz,
                geojson=geo_district_data_viz.geometry,
                locations=geo_district_data_viz.index,
                color="poblacion_total",
                center={"lat": 9.93, "lon": -84.08},
                zoom=7,
                opacity=0.5,
                title="Grafo de Contigüidad de Distritos sobre Mapa de Población",
            )

            fig.update_traces(
                hoverinfo="none",
                hovertemplate=None,
                selector=dict(type="choroplethmap"),
            )

            fig.update_layout(
                map_style="carto-positron",  # This style works with MapLibre
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                legend_title_text="Población",
            )
            return fig

        # --- 2. Generate the Geographic Layout for the Graph (no changes here) ---
        geo_data_for_layout = geo_district_data.to_crs(epsg=4326)
        pos = {
            node: (geom.centroid.x, geom.centroid.y)
            for node, geom in geo_data_for_layout.geometry.items()
        }

        # --- 3. Create the Edge Layer ---
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scattermap(
            lon=edge_x,
            lat=edge_y,
            mode="lines",
            line=dict(width=1, color="#FFFFFF"),
            hoverinfo="none",
            showlegend=False,
        )

        # --- 4. Create the Node Layer with Rich Tooltip Information ---
        node_lon, node_lat, hover_texts, custom_data = [], [], [], []
        for node in G.nodes():
            lon, lat = pos[node]
            node_lon.append(lon)
            node_lat.append(lat)
            info = geo_data_for_layout.loc[node]
            hover_texts.append(info["distrito"])
            custom_data.append(
                [
                    info["provincia"],
                    info["poblacion_total"],
                    info["area_km2"],
                ],
            )

        node_trace = go.Scattermap(
            lon=node_lon,
            lat=node_lat,
            text=hover_texts,
            customdata=custom_data,
            mode="markers",
            marker=dict(size=9, color="black", opacity=0.9),
            showlegend=False,
            hovertemplate=(
                "<b>%{text}</b><br><br>"
                + "<b>Provincia:</b> %{customdata[0]}<br>"
                + "<b>Población:</b> %{customdata[1]:,}<br>"
                + "<b>Área:</b> %{customdata[2]:.2f} km²"
                + "<extra></extra>"
            ),
        )

        # --- 5. Combine the Map and the Graph Layers (no changes here) ---
        fig = create_base_map()
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)

        # --- 6. Show the Final Interactive Visualization ---
        return fig.show()


    make_graph_map()
    return


@app.cell
def show_interactive_map(geo_district_data, px):
    def make_interactive_map():
        geo_district_data_viz = geo_district_data.to_crs(epsg=4326)

        fig = px.choropleth_map(
            geo_district_data_viz,
            geojson=geo_district_data_viz.geometry,
            locations=geo_district_data_viz.index,
            color="poblacion_total",
            hover_name="distrito",
            hover_data={
                "provincia": True,
                "poblacion_total": ":,",
                "area_km2": ":.2f",
            },
            center={
                "lat": 9.93,
                "lon": -84.08,
            },
            zoom=7,
            opacity=0.7,
            title="Población de Distritos de Costa Rica",
        )

        # Customize the layout for a cleaner look
        fig.update_layout(
            margin={
                "r": 0,
                "t": 40,
                "l": 0,
                "b": 0,
            },
            legend_title_text="Población",
        )

        # To show the figure in a notebook or script
        return fig.show()


    make_interactive_map()
    return


if __name__ == "__main__":
    app.run()
