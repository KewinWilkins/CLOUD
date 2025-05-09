import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import warnings
from prophet import Prophet

warnings.simplefilter(action="ignore", category=FutureWarning)

file_path = "AdidasUSSalesDatasets.xlsx"
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name="Data Sales Adidas")

df = df.iloc[3:].reset_index(drop=True)
df.columns = ["Index", "Retailer", "Retailer ID", "Invoice Date", "Region", "State", "City",
              "Product", "Price per Unit", "Units Sold", "Total Sales", "Operating Profit",
              "Operating Margin", "Sales Method"]
df = df.iloc[1:].reset_index(drop=True)
df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")
df.drop(columns=["Index"], inplace=True)

df.dropna(subset=["Invoice Date", "Units Sold", "Total Sales", "Price per Unit"], inplace=True)

df = df[df["Units Sold"] > 0]

df = df[df["Total Sales"] > 0]

app = dash.Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"])

conn = sqlite3.connect("sales_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS past_views (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_date TEXT,
        end_date TEXT,
        selected_regions TEXT,
        selected_products TEXT
    )
""")
conn.commit()

app.layout = html.Div(className="container mt-4", children=[
    html.H1("Adidas Sales Dashboard", className="text-center mb-4 text-primary"),

    html.Div(className="row mb-3", children=[
        html.Div(className="col-md-4", children=[
            html.Label("Select Date Range", className="fw-bold"),
            dcc.DatePickerRange(
                id="date-picker",
                start_date=df["Invoice Date"].min(),
                end_date=df["Invoice Date"].max(),
                display_format="YYYY-MM-DD",
                className="form-control"
            )
        ]),
        html.Div(className="col-md-4", children=[
            html.Label("Select Region", className="fw-bold"),
            dcc.Dropdown(
                id="region-dropdown",
                options=[{"label": i, "value": i} for i in df["Region"].dropna().unique()],
                placeholder="All Regions",
                multi=True,
                className="form-select"
            )
        ]),
        html.Div(className="col-md-4", children=[
            html.Label("Select Product", className="fw-bold"),
            dcc.Dropdown(
                id="product-dropdown",
                options=[{"label": i, "value": i} for i in df["Product"].dropna().unique()],
                placeholder="All Products",
                multi=True,
                className="form-select"
            )
        ]),
    ]),

    html.Div(className="row text-white text-center", children=[
        html.Div(className="col-md-4", children=[
            html.Div("Total Sales: $0", id="total-sales", className="card bg-primary p-3")
        ]),
        html.Div(className="col-md-4", children=[
            html.Div("Units Sold: 0", id="total-units", className="card bg-success p-3")
        ]),
        html.Div(className="col-md-4", children=[
            html.Div("Avg Price: $0", id="avg-price", className="card bg-warning p-3")
        ])
    ]),

    html.Div(className="row mt-4", children=[
        html.Div(className="col-md-6", children=[
            dcc.Graph(id="sales-trend")
        ]),
        html.Div(className="col-md-6", children=[
            dcc.Graph(id="sales-pie-chart")
        ]),
    ]),

    html.Div(className="row mt-4", children=[
        html.Div(className="col-md-12", children=[
            dcc.Graph(id="forecast-graph")
        ])
    ]),

    html.Div(className="text-center mt-4", children=[
        html.Button("Save View", id="save-view-btn", className="btn btn-primary", n_clicks=0),
        html.Button("View Past Data", id="view-past-btn", className="btn btn-secondary", n_clicks=0),
        html.Button("Delete All", id="delete-all-btn", className="btn btn-danger", n_clicks=0),
    ]),

    html.Div(id="past-data-output", className="mt-4")
])

@app.callback(
    [Output("sales-trend", "figure"),
     Output("sales-pie-chart", "figure"),
     Output("forecast-graph", "figure"),
     Output("total-sales", "children"),
     Output("total-units", "children"),
     Output("avg-price", "children")],
    [Input("date-picker", "start_date"),
     Input("date-picker", "end_date"),
     Input("region-dropdown", "value"),
     Input("product-dropdown", "value")]
)
def update_dashboard(start_date, end_date, selected_regions, selected_products):
    filtered_df = df[(df["Invoice Date"] >= start_date) & (df["Invoice Date"] <= end_date)]
    if selected_regions:
        filtered_df = filtered_df[filtered_df["Region"].isin(selected_regions)]
    if selected_products:
        filtered_df = filtered_df[filtered_df["Product"].isin(selected_products)]
    if filtered_df.empty:
        return px.line(title="No Data"), px.pie(title="No Data"), px.line(title="No Data"), "Total Sales: $0", "Units Sold: 0", "Avg Price: $0"
    
    sales_agg = filtered_df.groupby("Invoice Date")["Units Sold"].sum().reset_index()
    sales_trend_fig = px.line(sales_agg, x="Invoice Date", y="Units Sold", title="Sales Trend Over Time")
    sales_pie_fig = px.pie(filtered_df, names="Product", values="Total Sales", title="Sales Breakdown by Product")
    
    forecast_model = Prophet()
    forecast_data = filtered_df.groupby("Invoice Date")["Units Sold"].sum().reset_index()
    forecast_data.columns = ["ds", "y"]
    forecast_model.fit(forecast_data)
    future = forecast_model.make_future_dataframe(periods=365)
    forecast = forecast_model.predict(future)
    forecast_fig = px.line(forecast, x="ds", y="yhat", title="Sales Forecast")
    return sales_trend_fig, sales_pie_fig, forecast_fig, f"Total Sales: ${filtered_df['Total Sales'].sum():,.0f}", f"Units Sold: {filtered_df['Units Sold'].sum():,.0f}", f"Avg Price: ${filtered_df['Price per Unit'].mean():.2f}"

if __name__ == "__main__":
    app.run(debug=False)
