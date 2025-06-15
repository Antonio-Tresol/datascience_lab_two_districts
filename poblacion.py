import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""#<center>Construcción del Grafo Poblacional de Costa Rica a nivel Distrital</center>"""
    )
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
    import numpy as np

    return go, gpd, mo, np, nx, pd, px


@app.cell
def open_data(gpd, mo, pd):
    district_population_data = pd.read_excel(
        mo.notebook_dir() / "legacy_notebooks/data/df_distritos.xlsx"
    )
    district_population_data["Codigo"] = district_population_data["Codigo"].astype(
        "str"
    )

    shapefile_path = (
        mo.notebook_dir() / "legacy_notebooks/data/UGED_MGN_2022/UGED_MGN_2022.shp"
    )

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
def _(geo_district_data, gpd, np):
    def polsby_popper(geom: gpd.GeoDataFrame) -> float:
        if geom and geom.area > 0 and geom.length > 0:
            return (4 * np.pi * geom.area) / (geom.length**2)
        return 0

    geo_district_data["polsby_popper"] = geo_district_data.geometry.apply(polsby_popper)
    geo_district_data[["distrito", "polsby_popper"]].sort_values(
        by="polsby_popper", ascending=True
    )

    geo_district_data.drop(columns="geometry")
    return


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

    # Métricas del grafo
    print(f"Número de nodos: {G.number_of_nodes()}")
    print(f"Número de aristas: {G.number_of_edges()}")
    grados = [d for _, d in G.degree()]
    print(f"Grado medio: {sum(grados) / len(grados):.2f}")
    print(f"Número de componentes conexas: {nx.number_connected_components(G)}")
    return (G,)


@app.cell
def _(G, nx):
    componentes = list(nx.connected_components(G))
    print(f"Número de componentes conexas: {len(componentes)}")
    if len(componentes) == 1:
        print("Todos los distritos están contiguamente conectados.")
    else:
        print("Advertencia: hay distritos desconectados.")
    return


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


@app.cell
def _(mo):
    mo.md(
        r"""
    # Para evaluar
    - Validar que el grafo de continuidad generado por el clustering mantenga los componenetes conexos preexistentes.
    - Entre menos desviación porcentual de población haya entre los distritos nuevos (creados por el clustering, mejor esta funcionan el clustering)
    - En polbsy poper cercano a uno es mejor
    """
    )
    return


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import KMeans

    return (KMeans,)


@app.cell
def _(geo_district_data):
    # Nuevo valor de k
    poblacion_total = geo_district_data["poblacion_total"].sum()
    meta_poblacional = 45000  # nueva cantidad
    cantidad_diputados = poblacion_total / meta_poblacional

    k = round(poblacion_total / meta_poblacional)
    print(f"Número estimado de distritos (k): {k}")
    return (k,)


@app.cell
def _(KMeans, geo_district_data, k, pd):
    coordinates = pd.DataFrame(
        data={
            "x": geo_district_data.geometry.centroid.x,
            "y": geo_district_data.geometry.centroid.y,
        },
    ).values
    coordinates

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(coordinates)
    new_geo_districts = geo_district_data.copy()

    new_geo_districts["distrito_nuevo"] = kmeans.labels_
    # TODO: hacer gráfico interactivo
    new_geo_districts.plot(
        column="distrito_nuevo",
        categorical=True,
        figsize=(20, 20),
        legend=False,
        edgecolor="k",
    )
    return


if __name__ == "__main__":
    app.run()
