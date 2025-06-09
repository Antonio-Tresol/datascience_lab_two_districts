import marimo

__generated_with = "0.13.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from openpyxl import load_workbook

    try:
        wb = load_workbook("data/repoblacev2011-2025-03_2.xlsx")
    except FileNotFoundError:
        print("Error: The file 'repoblacev2011-2025-03_2.xlsx' was not found.")
        print("Please upload the file and run the script again.")
    else:
        # Select the worksheet named "2022"
        if "2025" in wb.sheetnames:
            ws = wb["2025"]

            # A list to store the rows with an indent of 2
            indented_rows = []

            # Iterate through the rows from 10 to 576
            for row_num in range(10, 577):
                # Get the cell in column B
                cell_b = ws[f"B{row_num}"]

                # Check the indent level
                indent_level = cell_b.alignment.indent
                if indent_level == 2:
                    # Get the value from column C in the same row
                    cell_c_value = ws[f"C{row_num}"].value
                    # Add the tuple of (value in B, value in C) to our list
                    indented_rows.append((cell_b.value, cell_c_value))

            # Convert the list of tuples to a tuple of tuples
            result_tuple = tuple(indented_rows)

            # Print the result
            print(result_tuple)
        else:
            print("Worksheet '2022' not found in the workbook.")
    return (result_tuple,)


@app.cell
def _(result_tuple):
    len(result_tuple)
    return


@app.cell
def _(result_tuple):
    import polars as pl

    dataset_poblacion = pl.DataFrame(
        data=result_tuple, schema=["distrito", "poblacion"], orient="row"
    )
    dataset_poblacion
    return


if __name__ == "__main__":
    app.run()
