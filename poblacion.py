import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import geopandas as gpd
    import marimo as mo
    return gpd, mo, pd


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
    return


if __name__ == "__main__":
    app.run()
