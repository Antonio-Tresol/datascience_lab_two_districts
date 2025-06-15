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
    import numpy as np
    return go, gpd, mo, np, nx, pd, px


@app.cell
def _():
    organizer = HierarchyOrganizer()
    return (organizer,)


@app.cell
def _(mo, organizer):
    organizer.new(level=1)
    mo.md(f"""## {organizer.format()}. **Cargado, limpieza y Exploración de los Datos**""")
    return


@app.cell
def open_data(gpd, mo, pd):
    district_path = mo.notebook_dir() / "data" / "df_distritos.xlsx"
    district_population_data = pd.read_excel(str(district_path))
    district_population_data["Codigo"] = district_population_data["Codigo"].astype("str")

    shapefile_path = mo.notebook_dir() / "data" / "UGED_MGN_2022" / "UGED_MGN_2022.shp"

    district_geospatial_data = gpd.read_file(str(shapefile_path))

    district_geospatial_data = district_geospatial_data[
        district_geospatial_data.geometry.notnull()
    ].copy()

    district_geospatial_data = district_geospatial_data.set_crs(epsg=5367)
    return district_geospatial_data, district_population_data


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
    return (district_data,)


@app.cell
def _(district_data, gpd, mo, organizer):
    geo_district_data = gpd.GeoDataFrame(district_data)

    organizer.new(level=2)
    title_original_data = mo.md(text=f"### {organizer.format()}. Datos sin la columna de geometría")

    mo.vstack(
        items=[
            title_original_data,
            geo_district_data.drop(columns="geometry"),
        ]
    )
    return (geo_district_data,)


@app.cell
def calculate_metrics(geo_district_data, mo, organizer, polsby_popper):
    """Calculate district metrics like Polsby-Popper compactness"""

    geo_district_data["polsby_popper"] = geo_district_data.geometry.apply(polsby_popper)
    geo_district_data[["distrito", "polsby_popper"]] = geo_district_data[
        ["distrito", "polsby_popper"]
    ].sort_values(by="polsby_popper", ascending=True)

    organizer.new(level=2)
    title_original_data_with_polsby_popper = mo.md(
        text=f"### {organizer.format()}. Distritos originales con polsby popper (sin la columna de geometría)"
    )

    mo.vstack(
        items=[
            title_original_data_with_polsby_popper,
            geo_district_data.drop(columns="geometry"),
        ]
    )
    return


@app.cell
def _(mo, organizer):
    organizer.new(level=2)
    mo.md(f"""### {organizer.format()} Grafo de contiguidad""")
    return


@app.cell
def make_neighbor_graph(build_continuity_graph, geo_district_data, nx):
    """Create the contiguity graph and display its metrics"""

    G = build_continuity_graph(geo_district_data)

    # Graph metrics
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
def plot_map_with_continuity_graph(G, geo_district_data, make_graph_map):
    """Cell for executing plotting functions"""

    # Display the contiguity graph
    make_graph_map(
        G, geo_district_data, title="Grafo de Contigüidad de Distritos sobre Mapa de Población"
    )
    return


@app.cell
def plot_map(geo_district_data, make_interactive_map):
    # Display the interactive population map
    make_interactive_map(
        geo_df=geo_district_data,
        title="Población de Distritos de Costa Rica",
        color_col="poblacion_total",
    )
    return


@app.cell
def _(mo, organizer):
    organizer.new(level=1)
    mo.md(
        f"""
    ## {organizer.format()}. Evaluación de modelos"""
    )
    return


@app.cell
def _(mo, organizer):
    organizer.new(level=2)
    mo.md(f"""### {organizer.format()}. Para evaluar
    - Validar que el grafo de continuidad generado por el clustering mantenga los componenetes conexos preexistentes.
    - Entre menos desviación porcentual de población haya entre los distritos nuevos (creados por el clustering, mejor esta funcionan el clustering)
    - En polbsy poper cercano a uno es mejor
    """)
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
def _(KMeans, geo_district_data, k, make_interactive_map, pd):
    coordinates = pd.DataFrame(
        data={
            "x": geo_district_data.geometry.centroid.x,
            "y": geo_district_data.geometry.centroid.y,
        },
    ).values

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(coordinates)
    new_geo_districts = geo_district_data.copy()

    new_geo_districts["distrito_nuevo"] = kmeans.labels_
    # TODO: hacer gráfico interactivo
    make_interactive_map(
        new_geo_districts,
        "Distritos Creados por KMeans",
        "distrito_nuevo",
        extra_hover_data={"distrito_nuevo": True},
    )
    return


@app.cell
def _(mo, organizer):
    organizer.new(level=1)
    mo.md(f"""## {organizer.format()}. Anexo""")
    return


@app.cell
def evaluation_metrics(np):
    def polsby_popper(geometry):
        """Calculate Polsby-Popper compactness metric for a geometry"""
        if geometry and geometry.area > 0 and geometry.length > 0:
            return (4 * np.pi * geometry.area) / (geometry.length**2)
        return 0
    return (polsby_popper,)


@app.cell
def define_plotting_functions(go, gpd, nx, px):
    """Define all utility functions"""


    def build_continuity_graph(geo_df: gpd.GeoDataFrame) -> nx.Graph:
        """Build a contiguity graph from geographic data"""
        G = nx.Graph()

        # Add nodes with population
        for idx, row in geo_df.iterrows():
            G.add_node(idx, nombre=row["distrito"], poblacion=row["poblacion_total"])

        # Add edges between contiguous districts
        for i, geom in geo_df.geometry.items():
            vecinos = geo_df[geo_df.geometry.touches(geom)].index
            for v in vecinos:
                if not G.has_edge(i, v):
                    G.add_edge(i, v)
        return G


    def make_graph_map(
        G: nx.Graph,
        geo_district_data: gpd.GeoDataFrame,
        title: str,
        color_col: str = "poblacion_total",
    ) -> None:
        """Create an interactive map showing the contiguity graph"""

        # --- 1. Create the Base Map (with hover disabled on polygons) ---
        def create_base_map() -> None:
            geo_district_data_viz = geo_district_data.to_crs(epsg=4326)

            fig = px.choropleth_map(
                geo_district_data_viz,
                geojson=geo_district_data_viz.geometry,
                locations=geo_district_data_viz.index,
                color=color_col,
                center={"lat": 9.93, "lon": -84.08},
                zoom=7,
                opacity=0.5,
                title=title,
            )

            fig.update_traces(
                hoverinfo="none",
                hovertemplate=None,
                selector=dict(type="choroplethmap"),
            )

            fig.update_layout(
                map_style="carto-positron",
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                legend_title_text="Población",
            )
            return fig

        # --- 2. Generate the Geographic Layout for the Graph ---
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

        # --- 5. Combine the Map and the Graph Layers ---
        fig = create_base_map()
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)

        # --- 6. Show the Final Interactive Visualization ---
        fig.show()


    def make_interactive_map(
        geo_df: gpd.GeoDataFrame,
        title: str,
        color_col: str,
        extra_hover_data: dict = {},
    ) -> None:
        """Create an interactive choropleth map"""
        geo_district_data_viz = geo_df.to_crs(epsg=4326)
        hover_data = {
            "provincia": True,
            "poblacion_total": ":,",
            "area_km2": ":.2f",
        }
        hover_data.update(extra_hover_data)
        fig = px.choropleth_map(
            geo_district_data_viz,
            geojson=geo_district_data_viz.geometry,
            locations=geo_district_data_viz.index,
            color=color_col,
            hover_name="distrito",
            hover_data=hover_data,
            center={
                "lat": 9.93,
                "lon": -84.08,
            },
            zoom=7,
            opacity=0.7,
            title=title,
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
        fig.show()
    return build_continuity_graph, make_graph_map, make_interactive_map


@app.class_definition(hide_code=True)
class HierarchyOrganizer:
    """
    A class to manage and track numbering for a hierarchy of any depth.

    This organizer can handle numbering schemes like "1", "1.1", "1.1.1", etc.,
    making it useful for complex documents, outlines, or task lists.
    """

    def __init__(self, depth: int = 3):
        """
        Initializes the organizer for a given hierarchy depth.

        Args:
            depth (int, optional): The number of levels in the hierarchy.
                                   Defaults to 3.
        """
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("Depth must be a positive integer.")
        self._depth = depth
        self._state = [0] * self._depth

    def new(self, level: int) -> list[int]:
        """
        Starts a new item at the specified level, resetting all lower levels.

        For example, calling new(1) for Section will create a new Section 1,
        and calling new(2) for Subsection will create a new Subsection 1.1.

        Args:
            level (int): The 1-based level to increment (e.g., 1 for section,
                         2 for subsection).

        Returns:
            list[int]: The new state of the hierarchy as a list of numbers.
        """
        if not (1 <= level <= self._depth):
            raise IndexError(f"Level must be between 1 and {self._depth}.")

        level_index = level - 1  # Convert to 0-based index

        # Increment the specified level
        self._state[level_index] += 1

        # Reset all subsequent (lower) levels to zero
        for i in range(level_index + 1, self._depth):
            self._state[i] = 0

        return self.state

    @property
    def state(self) -> list[int]:
        """Returns the current hierarchy state as a list of numbers."""
        return list(self._state)

    def format(self) -> str:
        """
        Formats the current state into a dot-separated string like "1.2.1".

        Trailing zeros are omitted for a clean, standard representation.
        For example, a state of [2, 1, 0] is formatted as "2.1".
        """
        # Find the last level that is not zero
        last_significant_index = -1
        for i in range(self._depth - 1, -1, -1):
            if self._state[i] != 0:
                last_significant_index = i
                break

        # If all levels are zero (initial state), return "0"
        if last_significant_index == -1:
            return "0"

        # Join the numbers up to the last significant level
        active_levels = self._state[: last_significant_index + 1]
        return ".".join(map(str, active_levels))

    def reset(self) -> None:
        """Resets the organizer back to its initial state."""
        self._state = [0] * self._depth


if __name__ == "__main__":
    app.run()
